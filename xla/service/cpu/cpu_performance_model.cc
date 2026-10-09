/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/cpu/cpu_performance_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/cpu_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/service/cpu/parallel_task_assignment.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

// Returns the number of bytes of `operand` that have to be read from memory at
// least once. CpuInstructionFusion always fuses broadcasts into their users, so
// only the input of a broadcast is read from memory.
int64_t GetOperandNetBytes(const CpuHloCostAnalysis* cost_analysis,
                           const HloInstruction* operand) {
  if (operand->opcode() == HloOpcode::kBroadcast) {
    return cost_analysis->GetShapeSize(operand->operand(0)->shape());
  }
  return cost_analysis->GetShapeSize(operand->shape());
}

}  // namespace

EstimateRunTimeData EstimateRunTimeData::Infinite() {
  return EstimateRunTimeData{
      /*flops=*/std::numeric_limits<int64_t>::max(),
      /*bytes_read=*/std::numeric_limits<int64_t>::max(),
      /*bytes_written=*/std::numeric_limits<int64_t>::max(),
      /*read_time=*/absl::InfiniteDuration(),
      /*write_time=*/absl::InfiniteDuration(),
      /*compute_time=*/absl::InfiniteDuration(),
      /*exec_time=*/absl::InfiniteDuration()};
}

std::string EstimateRunTimeData::ToString() const {
  return absl::StrFormat(
      "EstimateRunTimeData{\n"
      " flops: %d\n"
      " bytes_read: %d\n"
      " bytes_written: %d\n"
      " read_time: %s\n"
      " write_time: %s\n"
      " compute_time: %s\n"
      " exec_time: %s\n"
      "}",
      flops, bytes_read, bytes_written, absl::FormatDuration(read_time),
      absl::FormatDuration(write_time), absl::FormatDuration(compute_time),
      absl::FormatDuration(exec_time));
}

/*static*/
se::DeviceDescription CpuPerformanceModel::DefaultDeviceInfo() {
  se::DeviceDescription device_info;
  const int threads_per_core =
      std::max(1, tsl::port::NumHyperthreadsPerCore());
  device_info.set_threads_per_core_limit(threads_per_core);
  device_info.set_core_count(
      std::max(1, tsl::port::MaxParallelism() / threads_per_core));
  // A core computes 19.5 f32 adds per ns in a vectorized loop fusion.
  device_info.set_clock_rate_ghz(3.25);
  device_info.set_fpus_per_core(6);
  // Memory bandwidth of all cores, reading and writing.
  device_info.set_memory_bandwidth(int64_t{65} * 1000 * 1000 * 1000);
  device_info.set_l2_cache_size(int64_t{512} << 10);
  return device_info;
}

EstimateRunTimeData CpuPerformanceModel::EstimateRunTimeForInstruction(
    const HloInstruction* instr,
    const CpuHloCostAnalysis* cost_analysis) const {
  int64_t flops = cost_analysis->flop_count(*instr);
  int64_t bytes_written = cost_analysis->output_bytes_accessed(*instr);
  // HloCostAnalysis reports negative values for instructions it cannot model,
  // e.g. custom calls.
  if (flops < 0 || bytes_written < 0) {
    return EstimateRunTimeData::Infinite();
  }

  absl::Duration compute_time = ComputeTime(device_info_, flops);

  absl::Duration read_time;
  int64_t bytes_read = 0;
  for (const HloInstruction* operand : instr->operands()) {
    int64_t operand_size = GetOperandNetBytes(cost_analysis, operand);
    int64_t n_bytes_total =
        GetOperandBytesAccessed(cost_analysis, instr, operand);
    if (n_bytes_total < 0) {
      return EstimateRunTimeData::Infinite();
    }
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
    const int64_t read_factor = ReadFactor(instr, operand);
    n_bytes_total *= read_factor;
    n_bytes_net *= read_factor;
    bytes_read += n_bytes_total;
    read_time +=
        ReadTimeWithDRAMHeuristic(device_info_, n_bytes_net, n_bytes_total);
  }

  absl::Duration write_time = WriteTime(device_info_, bytes_written);
  const int64_t cores =
      KernelCores(*instr, flops, cost_analysis->transcendental_flop_count(*instr),
                  bytes_read + bytes_written);
  compute_time = ScaleToCores(compute_time, cores, &read_time, &write_time);
  absl::Duration exec_time =
      CombineComputeAndMemoryAccessTime(compute_time, read_time + write_time);

  EstimateRunTimeData runtime_data = {flops,     bytes_read, bytes_written,
                                      read_time, write_time, compute_time,
                                      exec_time};
  VLOG(3) << "Runtime data for HLO: " << instr->name() << "\n"
          << runtime_data.ToString();
  return runtime_data;
}

absl::Duration CpuPerformanceModel::EstimateRunTimeForFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const CpuHloCostAnalysis* cost_analysis) const {
  if (producer_runtime.IsInfinite() || consumer_runtime.IsInfinite()) {
    return absl::InfiniteDuration();
  }

  float utilization_by_this_consumer =
      GetOperandUtilization(cost_analysis, consumer, producer);

  int64_t flops = std::llround(producer_runtime.flops *
                               utilization_by_this_consumer) +
                  consumer_runtime.flops;
  absl::Duration compute_time = ComputeTime(device_info_, flops);

  // Operands of the fused instruction: the operands of the producer and the
  // operands of the consumer other than the producer.
  absl::flat_hash_set<const HloInstruction*> fusion_operands;
  for (const HloInstruction* operand : producer->operands()) {
    fusion_operands.insert(operand);
  }
  for (const HloInstruction* operand : consumer->operands()) {
    if (operand != producer) {
      fusion_operands.insert(operand);
    }
  }

  absl::Duration read_time;
  int64_t bytes_read = 0;
  for (const HloInstruction* operand : fusion_operands) {
    int64_t operand_size = GetOperandNetBytes(cost_analysis, operand);
    int64_t n_bytes_total = GetSharedOperandBytesAccessed(
        cost_analysis, producer, consumer, operand);
    if (n_bytes_total < 0) {
      return absl::InfiniteDuration();
    }
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
    const int64_t read_factor = ReadFactor(consumer, operand);
    n_bytes_total *= read_factor;
    n_bytes_net *= read_factor;
    bytes_read += n_bytes_total;
    read_time +=
        ReadTimeWithDRAMHeuristic(device_info_, n_bytes_net, n_bytes_total);
  }

  // The fused kernel computes the consumer's output.
  absl::Duration write_time = WriteTime(device_info_,
                                        consumer_runtime.bytes_written);
  const int64_t cores = KernelCores(
      *consumer, flops,
      std::llround(cost_analysis->transcendental_flop_count(*producer) *
                   utilization_by_this_consumer) +
          cost_analysis->transcendental_flop_count(*consumer),
      bytes_read + consumer_runtime.bytes_written);
  compute_time = ScaleToCores(compute_time, cores, &read_time, &write_time);
  absl::Duration exec_time =
      CombineComputeAndMemoryAccessTime(compute_time, read_time + write_time);

  VLOG(3) << "Runtime data for producer-consumer fusion:\n"
          << " producer: " << producer->name() << "\n"
          << " consumer: " << consumer->name() << "\n"
          << EstimateRunTimeData{flops,
                                 bytes_read,
                                 consumer_runtime.bytes_written,
                                 read_time,
                                 write_time,
                                 compute_time,
                                 exec_time}
                 .ToString();
  return exec_time;
}

CpuPerformanceModel::RunTimes CpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const CpuHloCostAnalysis* cost_analysis,
    absl::Span<const HloInstruction* const> fused_consumers) const {
  EstimateRunTimeData producer_runtime =
      EstimateRunTimeForInstruction(producer, cost_analysis);

  // Fusing `producer` into all of its users removes its kernel.
  absl::Duration time_unfused = producer_runtime.exec_time;
  if (!producer_runtime.IsInfinite()) {
    time_unfused += KernelOverhead(LoopFusionTasks(
        producer_runtime.flops,
        cost_analysis->transcendental_flop_count(*producer),
        producer_runtime.bytes_read + producer_runtime.bytes_written));
    time_unfused +=
        MaterializeFaultTime(*producer, producer_runtime.bytes_written);
  }
  absl::Duration time_fused;

  for (const HloInstruction* fused_consumer : fused_consumers) {
    EstimateRunTimeData consumer_runtime =
        EstimateRunTimeForInstruction(fused_consumer, cost_analysis);
    time_unfused += consumer_runtime.exec_time;
    time_fused +=
        EstimateRunTimeForFusion(producer, fused_consumer, producer_runtime,
                                 consumer_runtime, cost_analysis);
  }

  VLOG(8) << "Producer: " << producer->name()
          << ", consumer count: " << fused_consumers.size()
          << ", unfused time: " << time_unfused
          << ", fused time: " << time_fused;
  return {time_unfused, time_fused};
}

/*static*/
int64_t CpuPerformanceModel::GetOperandBytesAccessed(
    const CpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  if (!instr->IsUserOf(operand)) {
    return 0;
  }
  return cost_analysis->operand_bytes_accessed(*instr,
                                               instr->operand_index(operand));
}

/*static*/
float CpuPerformanceModel::GetOperandUtilization(
    const CpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  float utilization = 0.f;
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    if (instr->operand(i) == operand) {
      utilization += cost_analysis->operand_utilization(*instr, i);
    }
  }
  return utilization;
}

namespace {

// Returns the utilization of `operand` that `producer` and `consumer` share
// after fusion. Covers the case where `producer` is elementwise and reads
// `operand` at the same indices as `consumer` does.
float GetCommonUtilization(const CpuHloCostAnalysis* cost_analysis,
                           const HloInstruction* producer,
                           const HloInstruction* consumer,
                           const HloInstruction* operand) {
  if (!producer->IsElementwise() || !consumer->IsUserOf(operand)) {
    return 0.f;
  }
  if (consumer->opcode() == HloOpcode::kFusion) {
    return cost_analysis->CommonElementwiseUtilization(
        consumer->fused_parameter(consumer->operand_index(operand)),
        consumer->fused_parameter(consumer->operand_index(producer)));
  }
  return consumer->IsElementwise() ? 1.f : 0.f;
}

}  // namespace

/*static*/
int64_t CpuPerformanceModel::GetSharedOperandBytesAccessed(
    const CpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
    const HloInstruction* consumer, const HloInstruction* operand) {
  float producer_utilization_by_consumer =
      GetOperandUtilization(cost_analysis, consumer, producer);

  int64_t bytes_accessed_by_producer =
      GetOperandBytesAccessed(cost_analysis, producer, operand);

  int64_t bytes_accessed_by_consumer =
      GetOperandBytesAccessed(cost_analysis, consumer, operand);

  if (bytes_accessed_by_producer < 0 || bytes_accessed_by_consumer < 0) {
    return -1;
  }

  float common_utilization =
      producer->IsUserOf(operand)
          ? GetCommonUtilization(cost_analysis, producer, consumer, operand)
          : 0.f;

  int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());
  int64_t common_bytes_accessed =
      std::llround(operand_size * common_utilization);

  return std::llround(bytes_accessed_by_producer *
                      producer_utilization_by_consumer) +
         bytes_accessed_by_consumer - common_bytes_accessed;
}

/*static*/
absl::Duration CpuPerformanceModel::ReadTimeWithDRAMHeuristic(
    const se::DeviceDescription& device_info, int64_t n_bytes_net,
    int64_t n_bytes_total) {
  // Loop fusions are emitted in layout order, so elements that are read more
  // than once, e.g. through a broadcast, are re-read by nearby iterations and
  // hit the cache. The first read hits the cache only if the whole operand
  // fits into it.
  float dram_bandwidth = device_info.memory_bandwidth();
  float cache_bandwidth = kCoreCacheBandwidth * device_info.core_count();
  if (n_bytes_net < device_info.l2_cache_size()) {
    dram_bandwidth = cache_bandwidth;
  }
  float rest_bandwidth = cache_bandwidth;
  int64_t n_bytes_read_dram = std::min(n_bytes_net, n_bytes_total);
  int64_t n_bytes_read_cache = n_bytes_total - n_bytes_read_dram;
  return absl::Seconds(n_bytes_read_dram / dram_bandwidth) +
         absl::Seconds(n_bytes_read_cache / rest_bandwidth);
}

/*static*/
absl::Duration CpuPerformanceModel::WriteTime(
    const se::DeviceDescription& device_info, int64_t bytes_written) {
  return absl::Seconds(1.0f * bytes_written / device_info.memory_bandwidth());
}

/*static*/
absl::Duration CpuPerformanceModel::ComputeTime(
    const se::DeviceDescription& device_info, int64_t flops) {
  // Flops are in units of an f32 add, see CpuHloCostAnalysis.
  double flops_per_ns = device_info.clock_rate_ghz() *
                        device_info.fpus_per_core() * device_info.core_count();
  return absl::Nanoseconds(1.0 * flops / flops_per_ns);
}

/*static*/
absl::Duration CpuPerformanceModel::CombineComputeAndMemoryAccessTime(
    absl::Duration compute_time, absl::Duration memory_access_time) {
  return compute_time + memory_access_time -
         std::min(compute_time, memory_access_time) * kMemoryComputeParallelism;
}

/*static*/
absl::Duration CpuPerformanceModel::ComputeTime(
    const se::DeviceDescription& device_info, int64_t flops,
    PrimitiveType type, int64_t threads) {
  // fpus_per_core counts f32 lanes; wider types fill fewer lanes.
  const int64_t bytes = primitive_util::ByteWidth(type);
  const double lanes =
      device_info.fpus_per_core() * 4.0 / std::max<int64_t>(4, bytes);
  const double flops_per_ns =
      device_info.clock_rate_ghz() * lanes * std::max<int64_t>(1, threads);
  return absl::Nanoseconds(1.0 * flops / flops_per_ns);
}

/*static*/
int64_t CpuPerformanceModel::ReadFactor(const HloInstruction* consumer,
                                        const HloInstruction* operand) {
  const HloInstruction* root = consumer->opcode() == HloOpcode::kFusion
                                   ? consumer->fused_expression_root()
                                   : consumer;
  if (root->opcode() != HloOpcode::kReduce || !operand->shape().IsArray()) {
    return 1;
  }
  const Shape& input = root->operand(0)->shape();
  if (ShapeUtil::ElementsIn(operand->shape()) != ShapeUtil::ElementsIn(input)) {
    return 1;
  }
  return StridedReadFactor(input, root->dimensions());
}


/*static*/
int64_t CpuPerformanceModel::StridedReadFactor(
    const Shape& input, absl::Span<const int64_t> reduced_dims) {
  if (!input.has_layout() || reduced_dims.empty()) {
    return 1;
  }
  const int64_t elem_bytes =
      std::max<int64_t>(1, primitive_util::ByteWidth(input.element_type()));
  // Bytes between consecutive elements of the innermost reduced dimension.
  int64_t stride = elem_bytes;
  for (int64_t dim : input.layout().minor_to_major()) {
    if (absl::c_linear_search(reduced_dims, dim)) {
      break;
    }
    stride *= input.dimensions(dim);
  }
  // Hardware prefetchers do not cross pages.
  return stride >= kPageBytes ? std::max<int64_t>(1, kCacheLineBytes / elem_bytes)
                              : 1;
}

/*static*/
absl::Duration CpuPerformanceModel::MemoryTime(
    const se::DeviceDescription& device_info, int64_t bytes,
    int64_t threads) {
  threads = std::max<int64_t>(1, threads);
  double bandwidth =
      bytes <= threads * device_info.l2_cache_size()
          ? threads * kCoreCacheBandwidth
          : std::min<double>(device_info.memory_bandwidth(),
                             threads * kCoreMemoryBandwidth);
  return absl::Seconds(bytes / bandwidth);
}

/*static*/
absl::Duration CpuPerformanceModel::ForkJoinTime(int64_t tasks) {
  return tasks > 1 ? kForkJoinTime + tasks * kForkJoinTaskTime
                   : absl::ZeroDuration();
}

/*static*/
absl::Duration CpuPerformanceModel::FreshAllocationTime(int64_t bytes) {
  return bytes > kFreshAllocationBytes
             ? kPageFaultTime * ((bytes + kPageFaultBytes - 1) / kPageFaultBytes)
             : absl::ZeroDuration();
}

absl::Duration CpuPerformanceModel::MaterializeFaultTime(
    const HloInstruction& producer, int64_t bytes) const {
  if (!RunsOncePerExecution(producer)) {
    return absl::ZeroDuration();
  }
  return temp_is_fresh_
             ? kPageFaultTime * ((bytes + kPageFaultBytes - 1) / kPageFaultBytes)
             : FreshAllocationTime(bytes);
}

/*static*/
bool CpuPerformanceModel::RunsOncePerExecution(const HloInstruction& instr) {
  const HloComputation* computation = instr.parent();
  return computation != nullptr && computation->IsEntryComputation();
}

absl::Duration CpuPerformanceModel::ReduceReadTime(int64_t bytes,
                                                   double cores) const {
  cores = std::max(1.0, cores);
  const double bandwidth =
      bytes <= cores * device_info_.l2_cache_size()
          ? cores * kCoreCacheBandwidth
          : std::min(kReduceReadBandwidth, cores * kCoreReduceReadBandwidth);
  return absl::Seconds(bytes / bandwidth);
}

/*static*/
absl::Duration CpuPerformanceModel::KernelOverhead(int64_t tasks) {
  return kKernelLaunchTime + ForkJoinTime(tasks);
}

int64_t CpuPerformanceModel::LoopFusionTasks(int64_t flops,
                                             int64_t transcendental_flops,
                                             int64_t bytes) const {
  // ParallelTaskAssigner counts each non-transcendental op as one flop and
  // transcendentals separately; an instruction with at most one flop per byte
  // runs on at most sqrt(threads) tasks.
  return DefaultParallelTaskCount(
      std::max<int64_t>(0, flops - transcendental_flops),
      /*transcendentals=*/0, bytes, Threads());
}

int64_t CpuPerformanceModel::Threads() const {
  return device_info_.core_count() *
         std::max<int64_t>(1, device_info_.threads_per_core_limit());
}

int64_t CpuPerformanceModel::KernelCores(const HloInstruction& kernel,
                                         int64_t flops,
                                         int64_t transcendental_flops,
                                         int64_t bytes) const {
  // The loop emitter partitions the outer dimensions of the kernel's output,
  // e.g. the outputs of a reduction.
  const Shape& shape = kernel.shape().IsTuple() &&
                               kernel.shape().tuple_shapes_size() > 0
                           ? kernel.shape().tuple_shapes(0)
                           : kernel.shape();
  const int64_t outputs =
      shape.IsArray() ? std::max<int64_t>(1, ShapeUtil::ElementsIn(shape)) : 1;
  return Cores(
      std::min(outputs, LoopFusionTasks(flops, transcendental_flops, bytes)));
}

absl::Duration CpuPerformanceModel::ScaleToCores(
    absl::Duration compute_time, int64_t cores, absl::Duration* read_time,
    absl::Duration* write_time) const {
  // ComputeTime, ReadTimeWithDRAMHeuristic and WriteTime assume all cores.
  const double all = device_info_.core_count();
  const double memory_scale =
      device_info_.memory_bandwidth() /
      std::min<double>(device_info_.memory_bandwidth(),
                       cores * kCoreMemoryBandwidth);
  *read_time = *read_time * memory_scale;
  *write_time = *write_time * memory_scale;
  return compute_time * (all / cores);
}

int64_t CpuPerformanceModel::Cores(int64_t tasks) const {
  return std::clamp<int64_t>(tasks, 1, device_info_.core_count());
}

namespace {

// Ops that a loop fusion computes element by element from its operands.
bool IsLoopFusible(const HloInstruction& instr) {
  return instr.IsElementwise() || instr.opcode() == HloOpcode::kBroadcast ||
         instr.opcode() == HloOpcode::kReshape ||
         instr.opcode() == HloOpcode::kBitcast ||
         instr.opcode() == HloOpcode::kTranspose ||
         instr.opcode() == HloOpcode::kSlice ||
         instr.opcode() == HloOpcode::kConcatenate ||
         instr.opcode() == HloOpcode::kPad ||
         instr.opcode() == HloOpcode::kReverse ||
         instr.opcode() == HloOpcode::kIota;
}

// Adds the time of one element of `instr` in elemental code to `work`.
// Concatenates and pads select their source per element.
void AddElementWork(const HloInstruction& instr, double weight,
                    CpuPerformanceModel::ChainWork* work) {
  int64_t flops = 0;
  if (instr.opcode() == HloOpcode::kConcatenate) {
    flops = instr.operand_count() - 1;
  } else if (instr.opcode() == HloOpcode::kPad) {
    flops = 1;
  } else if (instr.IsElementwise() && instr.opcode() != HloOpcode::kConvert &&
             instr.opcode() != HloOpcode::kCopy) {
    const PrimitiveType type = instr.shape().element_type();
    flops = CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(
        type, instr.opcode());
    if (HloCostAnalysis::IsTranscendental(instr.opcode())) {
      work->transcendental_flops += weight * flops;
      // A library fusion evaluates the transcendentals it supports with
      // vectorized polynomials, which cost about as much as an exp. Its sqrt
      // is slower, measured in a reduction of sqrt(d*d + 1) from broadcasts:
      // 0.35 ns f32, 1.03 ns f64 per element on one core.
      const bool is_sqrt = instr.opcode() == HloOpcode::kSqrt ||
                           instr.opcode() == HloOpcode::kRsqrt;
      const double library_flops =
          weight *
          (is_sqrt ? (primitive_util::ByteWidth(type) >= 8 ? 10 : 7)
                   : std::min(flops,
                              CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(
                                  type, HloOpcode::kExp)));
      work->library_flops += library_flops;
      work->library_transcendental_flops += library_flops;
      work->flops += weight * flops;
      return;
    }
  }
  work->flops += weight * flops;
  work->library_flops += weight * flops;
}

// Sums the per-element work of the loop-fusible chain rooted at `roots`,
// stopping at `stop`, and the bytes of the chain's leaves. A fused loop
// computes instructions shared by several roots once per element, and the
// operand of a broadcast once for the unrolled minor loops it adds. Leaves
// read only through broadcasts are reused from the cache; the others are
// streamed.
CpuPerformanceModel::ChainWork WalkChain(
    absl::Span<const HloInstruction* const> roots,
    absl::Span<const HloInstruction* const> stops) {
  CpuPerformanceModel::ChainWork work;
  // Instructions visited at all, and visited other than through a broadcast.
  absl::flat_hash_set<const HloInstruction*> seen;
  absl::flat_hash_set<const HloInstruction*> seen_streamed;
  // Instruction, whether it is reached through a broadcast, and the share of
  // its elements computed per element of the roots.
  std::vector<std::tuple<const HloInstruction*, bool, double>> stack;
  for (const HloInstruction* root : roots) {
    stack.push_back({root, /*via_broadcast=*/false, /*weight=*/1.0});
  }
  while (!stack.empty()) {
    auto [instr, via_broadcast, weight] = stack.back();
    stack.pop_back();
    if (via_broadcast ? seen.contains(instr)
                      : !seen_streamed.insert(instr).second) {
      continue;
    }
    const bool first = seen.insert(instr).second;
    const bool leaf =
        absl::c_linear_search(stops, instr) || !instr->shape().IsArray() ||
        !IsLoopFusible(*instr) ||
        (instr->opcode() == HloOpcode::kConstant &&
         !ShapeUtil::IsEffectiveScalar(instr->shape()));
    if (leaf) {
      if (instr->shape().IsArray()) {
        const int64_t bytes = ShapeUtil::ByteSizeOfElements(instr->shape());
        work.leaf_bytes += first ? bytes : 0;
        work.streamed_bytes += via_broadcast ? 0 : bytes;
      }
      continue;
    }
    if (first) {
      AddElementWork(*instr, weight, &work);
      // A library fusion reads a slice under a broadcast per element of the
      // broadcast, measured as 0.05 ns f32, 0.06 ns f64 per element for a
      // column of an [N,3] array.
      if (via_broadcast && instr->opcode() == HloOpcode::kSlice) {
        const double cost =
            weight *
            (primitive_util::ByteWidth(instr->shape().element_type()) >= 8 ? 0.6
                                                                          : 1);
        work.library_flops += cost;
        work.library_transcendental_flops += cost;
      }
    }
    const bool broadcast = instr->opcode() == HloOpcode::kBroadcast;
    const double operand_weight =
        broadcast ? weight / CpuHloCostAnalysis::UnrolledBroadcastElements(*instr)
                  : weight;
    for (const HloInstruction* operand : instr->operands()) {
      stack.push_back({operand, via_broadcast || broadcast, operand_weight});
    }
  }
  return work;
}

}  // namespace

std::string CpuPerformanceModel::ReduceWork::ToString() const {
  return absl::StrFormat(
      "ReduceWork{elements: %d, outputs: %d, flops/elem: %g, library "
      "flops/elem: %g, transcendental flops/elem: %g, leaf bytes: %d, "
      "streamed bytes: %d, materialized bytes: %d, materialized flops/elem: "
      "%g, strided read factor: %d, type: %s}",
      input_elements, outputs, chain.flops, chain.library_flops,
      chain.transcendental_flops, chain.leaf_bytes, chain.streamed_bytes,
      materialized_bytes, materialized.flops, strided_read_factor,
      primitive_util::LowercasePrimitiveTypeName(type));
}

/*static*/
CpuPerformanceModel::ReduceWork CpuPerformanceModel::AnalyzeReduce(
    const HloInstruction& reduce, const HloInstruction* materialized,
    bool materialized_anyway,
    absl::Span<const HloInstruction* const> also_materialized) {
  ReduceWork work;
  const Shape& input = reduce.operand(0)->shape();
  const Shape& output = reduce.shape().IsTuple()
                            ? reduce.shape().tuple_shapes(0)
                            : reduce.shape();
  work.input_elements = ShapeUtil::ElementsIn(input);
  work.outputs = ShapeUtil::ElementsIn(output);
  work.type = input.element_type();
  const int64_t num_inputs = reduce.operand_count() / 2;
  absl::Span<const HloInstruction* const> inputs =
      absl::MakeConstSpan(reduce.operands()).subspan(0, num_inputs);
  work.materialized_anyway = materialized != nullptr && materialized_anyway;
  std::vector<const HloInstruction*> loop_stops;
  if (work.materialized_anyway) {
    loop_stops.push_back(materialized);
  }
  work.chain = WalkChain(inputs, loop_stops);
  // The library fusion computes the chain above the materialized inputs.
  std::vector<const HloInstruction*> library_stops(also_materialized.begin(),
                                                   also_materialized.end());
  if (materialized != nullptr) {
    library_stops.push_back(materialized);
  }
  const ChainWork library = WalkChain(inputs, library_stops);
  work.chain.library_flops = library.library_flops;
  work.chain.library_transcendental_flops =
      library.library_transcendental_flops;
  // One reducer op per input element and operand.
  work.chain.flops += num_inputs;
  work.chain.library_flops += num_inputs;
  if (materialized != nullptr) {
    work.materialized = WalkChain({materialized}, /*stops=*/{});
    work.materialized_bytes =
        ShapeUtil::ByteSizeOfElements(materialized->shape());
    work.materialized_elements = ShapeUtil::ElementsIn(materialized->shape());
  }
  // Other inputs that the library fusion cannot absorb are written by loop
  // fusions too, and read by the library fusion.
  for (const HloInstruction* other : also_materialized) {
    const ChainWork other_work = WalkChain({other}, /*stops=*/{});
    const int64_t elements = ShapeUtil::ElementsIn(other->shape());
    work.extra_materialized_flops += std::llround(other_work.flops * elements);
    work.extra_materialized_transcendental_flops +=
        std::llround(other_work.transcendental_flops * elements);
    work.extra_materialized_bytes +=
        ShapeUtil::ByteSizeOfElements(other->shape());
    work.extra_materialized_leaf_bytes += other_work.leaf_bytes;
  }
  // The loop emitter vectorizes reductions only with reassociable reducers;
  // min and max keep the whole fused loop scalar.
  const HloInstruction* reducer_root = reduce.to_apply()->root_instruction();
  if (reducer_root->opcode() == HloOpcode::kTuple &&
      reducer_root->operand_count() > 0) {
    reducer_root = reducer_root->operand(0);
  }
  work.loop_reducer_vectorizes =
      reducer_root->opcode() != HloOpcode::kMinimum &&
      reducer_root->opcode() != HloOpcode::kMaximum;
  if (reduce.opcode() == HloOpcode::kReduce) {
    work.strided_read_factor = StridedReadFactor(input, reduce.dimensions());
    const int64_t minor_dim = input.has_layout()
                                  ? LayoutUtil::Minor(input.layout(), 0)
                                  : input.dimensions().size() - 1;
    work.reduces_minor_dim =
        absl::c_linear_search(reduce.dimensions(), minor_dim);
  }
  work.once_per_execution = RunsOncePerExecution(reduce);
  return work;
}

int64_t CpuPerformanceModel::LoopFusionTasks(const ReduceWork& work) const {
  const int64_t elem_bytes =
      std::max<int64_t>(1, primitive_util::ByteWidth(work.type));
  return LoopFusionTasks(
      std::llround(work.chain.flops * work.input_elements),
      std::llround(work.chain.transcendental_flops * work.input_elements),
      work.chain.leaf_bytes + work.outputs * elem_bytes);
}

bool CpuPerformanceModel::LoopFusionSplitsReduce(const ReduceWork& work) const {
  // TreeReductionRewriter's window: shorter reductions are not split.
  constexpr int64_t kTreeReductionWindow = 32;
  return work.outputs < LoopFusionTasks(work) &&
         work.input_elements >= kTreeReductionWindow * work.outputs;
}

CpuPerformanceModel::ReduceRunTimes CpuPerformanceModel::EstimateReduce(
    const ReduceWork& work) const {
  const int64_t elem_bytes =
      std::max<int64_t>(1, primitive_util::ByteWidth(work.type));
  ReduceRunTimes times;

  // Loop fusion. It is partitioned across outputs; TreeReductionRewriter
  // splits reductions to fewer outputs than tasks into a partitioned
  // reduce-window and a small reduction.
  {
    int64_t tasks = LoopFusionTasks(work);
    const bool split = LoopFusionSplitsReduce(work);
    if (!split) {
      tasks = std::min(tasks, std::max<int64_t>(1, work.outputs));
    }
    const int64_t cores = Cores(tasks);
    const int64_t read_bytes =
        work.chain.leaf_bytes +
        work.chain.streamed_bytes * (work.strided_read_factor - 1);
    // Measured with a 5-op chain reduced on one core: min costs 2.35x (f32)
    // and 1.64x (f64) as much as add.
    const double scalar_slowdown =
        work.loop_reducer_vectorizes ? 1.0 : (elem_bytes >= 8 ? 1.64 : 2.35);
    times.loop_fusion =
        CombineComputeAndMemoryAccessTime(
            ComputeTime(device_info_,
                        std::llround(work.chain.flops * work.input_elements *
                                     scalar_slowdown),
                        work.type, cores),
            ReduceReadTime(read_bytes, cores)) +
        KernelOverhead(tasks);
    if (split) {
      times.loop_fusion += kKernelLaunchTime;
    }
  }

  // Library fusion, after a loop fusion writes the materialized input.
  absl::Duration materialize = absl::ZeroDuration();
  {
    const bool primary = work.materialized_bytes > 0 && !work.materialized_anyway;
    const int64_t flops =
        (primary ? std::llround(work.materialized.flops *
                                work.materialized_elements)
                 : 0) +
        work.extra_materialized_flops;
    const int64_t transcendental_flops =
        (primary ? std::llround(work.materialized.transcendental_flops *
                                work.materialized_elements)
                 : 0) +
        work.extra_materialized_transcendental_flops;
    const int64_t written =
        (primary ? work.materialized_bytes : 0) + work.extra_materialized_bytes;
    const int64_t leaves =
        (primary ? work.materialized.leaf_bytes : 0) +
        work.extra_materialized_leaf_bytes;
    if (written > 0) {
      const int64_t tasks =
          LoopFusionTasks(flops, transcendental_flops, leaves + written);
      const int64_t cores = Cores(tasks);
      materialize =
          CombineComputeAndMemoryAccessTime(
              ComputeTime(device_info_, flops, work.type, cores),
              MemoryTime(device_info_, leaves + written, cores)) +
          KernelOverhead(tasks) +
          (work.once_per_execution ? FreshAllocationTime(written)
                                   : absl::ZeroDuration());
    }
  }
  {
    const int64_t reduced = std::max<int64_t>(
        1, work.input_elements / std::max<int64_t>(1, work.outputs));
    const int64_t tile = std::max<int64_t>(1, kLibraryReduceTileBytes /
                                                  elem_bytes);
    const double max_cores =
        std::max(1.0, kLibraryParallelEfficiency * device_info_.core_count());
    int64_t tasks;
    double cores;
    if (reduced > tile) {
      // Partial reductions of each output's tiles.
      const int64_t tiles = (reduced + tile - 1) / tile;
      tasks = tiles * work.outputs;
      cores = std::min(
          max_cores,
          std::max(1.0, 1.0 * tiles / kLibraryPartialReduceTilesPerThread));
    } else {
      tasks = (work.input_elements + tile - 1) / tile;
      cores = std::min<double>(max_cores, tasks);
    }
    // If it is materialized anyway, the loop fusion's leaves include it.
    const int64_t library_bytes =
        (work.materialized_anyway
             ? work.chain.leaf_bytes
             : work.chain.leaf_bytes + work.materialized_bytes) +
        work.extra_materialized_bytes;
    const double speedup = !work.reduces_minor_dim ? 1.0
                           : primitive_util::ByteWidth(work.type) >= 8
                               ? kLibraryArithmeticSpeedupF64
                               : kLibraryArithmeticSpeedupF32;
    const double library_flops =
        work.chain.library_transcendental_flops +
        (work.chain.library_flops - work.chain.library_transcendental_flops) /
            speedup;
    const absl::Duration compute =
        ComputeTime(device_info_,
                    std::llround(library_flops * work.input_elements),
                    work.type, /*threads=*/1) /
        cores;
    times.library =
        materialize +
        CombineComputeAndMemoryAccessTime(
            compute, ReduceReadTime(library_bytes, cores)) +
        kLibraryCallTime + ForkJoinTime(std::min(tasks, Threads()));
  }
  return times;
}

}  // namespace xla::cpu
