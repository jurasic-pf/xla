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
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/cpu_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
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

absl::Duration CpuPerformanceModel::ForkJoinTime(int64_t tasks) {
  return tasks > 1 ? kForkJoinTime + tasks * kForkJoinTaskTime
                   : absl::ZeroDuration();
}

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

}  // namespace xla::cpu
