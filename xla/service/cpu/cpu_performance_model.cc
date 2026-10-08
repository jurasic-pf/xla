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
#include <cstdlib>
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
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
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
  device_info.set_core_count(tsl::port::MaxParallelism());
  // 16 f32 FMA lanes per core: two 256-bit FMA units.
  device_info.set_fpus_per_core(16);
  device_info.set_clock_rate_ghz(3.0);
  device_info.set_memory_bandwidth(int64_t{100} * 1000 * 1000 * 1000);
  device_info.set_l2_cache_size(int64_t{1} << 20);
  // Experiment only: XLA_CPU_MODEL_GBPS / XLA_CPU_MODEL_GFLOPS override the
  // memory bandwidth and peak compute (keeping core count and clock).
  if (const char* env = std::getenv("XLA_CPU_MODEL_GBPS"); env != nullptr) {
    device_info.set_memory_bandwidth(
        static_cast<int64_t>(std::atof(env) * 1e9));
  }
  if (const char* env = std::getenv("XLA_CPU_MODEL_GFLOPS"); env != nullptr) {
    device_info.set_fpus_per_core(std::max(
        1, static_cast<int>(std::atof(env) /
                            (2 * 3.0 * device_info.core_count()))));
  }
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
    bytes_read += n_bytes_total;
    read_time +=
        ReadTimeWithDRAMHeuristic(device_info_, n_bytes_net, n_bytes_total);
  }

  absl::Duration write_time = WriteTime(device_info_, bytes_written);
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
    bytes_read += n_bytes_total;
    read_time +=
        ReadTimeWithDRAMHeuristic(device_info_, n_bytes_net, n_bytes_total);
  }

  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, read_time + consumer_runtime.write_time);

  VLOG(3) << "Runtime data for producer-consumer fusion:\n"
          << " producer: " << producer->name() << "\n"
          << " consumer: " << consumer->name() << "\n"
          << EstimateRunTimeData{flops,
                                 bytes_read,
                                 consumer_runtime.bytes_written,
                                 read_time,
                                 consumer_runtime.write_time,
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

  absl::Duration time_unfused = producer_runtime.exec_time;
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
  float cache_bandwidth = dram_bandwidth * kCacheSpeedup;
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
  double flops_per_ns = device_info.clock_rate_ghz() * /*fma:*/ 2 *
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
  // fpus_per_core counts f32 lanes.
  const int64_t bytes = primitive_util::ByteWidth(type);
  const double lanes = std::max(
      1.0, device_info.fpus_per_core() * 4.0 / std::max<int64_t>(4, bytes));
  const double flops_per_ns = device_info.clock_rate_ghz() * /*fma:*/ 2 *
                              lanes * std::max<int64_t>(1, threads);
  return absl::Nanoseconds(1.0 * flops / flops_per_ns);
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

// Flops per element of `instr` in elemental code. Concatenates and pads
// select their source per element.
int64_t FlopsPerElement(const CpuHloCostAnalysis& costs,
                        const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kConcatenate) {
    return instr.operand_count() - 1;
  }
  if (instr.opcode() == HloOpcode::kPad) {
    return 1;
  }
  if (!instr.IsElementwise() || instr.opcode() == HloOpcode::kConvert ||
      instr.opcode() == HloOpcode::kCopy) {
    return 0;
  }
  return costs.GetFlopsPerElementwiseOpElement(instr.shape().element_type(),
                                                instr.opcode());
}

// Sums the per-element flops of the loop-fusible chain rooted at `root`,
// stopping at `stop`, and the bytes of the chain's leaves.
void WalkChain(const CpuHloCostAnalysis& costs, const HloInstruction* root,
               const HloInstruction* stop, int64_t* flops_per_element,
               int64_t* leaf_bytes) {
  absl::flat_hash_set<const HloInstruction*> seen;
  std::vector<const HloInstruction*> stack = {root};
  while (!stack.empty()) {
    const HloInstruction* instr = stack.back();
    stack.pop_back();
    if (!seen.insert(instr).second) {
      continue;
    }
    if (instr == stop || !instr->shape().IsArray() || !IsLoopFusible(*instr) ||
        (instr->opcode() == HloOpcode::kConstant &&
         !ShapeUtil::IsEffectiveScalar(instr->shape()))) {
      if (instr->shape().IsArray()) {
        *leaf_bytes += ShapeUtil::ByteSizeOfElements(instr->shape());
      }
      continue;
    }
    *flops_per_element += FlopsPerElement(costs, *instr);
    for (const HloInstruction* operand : instr->operands()) {
      stack.push_back(operand);
    }
  }
}

}  // namespace

/*static*/
CpuPerformanceModel::ReduceWork CpuPerformanceModel::AnalyzeReduce(
    const HloInstruction& reduce, const HloInstruction* materialized) {
  CpuHloCostAnalysis costs(HloCostAnalysis::Options{});
  ReduceWork work;
  const Shape& input = reduce.operand(0)->shape();
  const Shape& output = reduce.shape().IsTuple()
                            ? reduce.shape().tuple_shapes(0)
                            : reduce.shape();
  work.input_elements = ShapeUtil::ElementsIn(input);
  work.outputs = ShapeUtil::ElementsIn(output);
  work.type = input.element_type();
  const int64_t num_inputs = reduce.operand_count() / 2;
  // One reducer op per input element and operand.
  int64_t flops_per_element = num_inputs;
  for (int64_t i = 0; i < num_inputs; ++i) {
    WalkChain(costs, reduce.operand(i), /*stop=*/nullptr, &flops_per_element,
              &work.leaf_bytes);
  }
  work.flops = flops_per_element * work.input_elements;
  if (materialized != nullptr) {
    int64_t materialized_flops_per_element = 0;
    int64_t unused_bytes = 0;
    WalkChain(costs, materialized, /*stop=*/nullptr,
              &materialized_flops_per_element, &unused_bytes);
    work.materialized_bytes =
        ShapeUtil::ByteSizeOfElements(materialized->shape());
    work.materialized_flops = materialized_flops_per_element *
                              ShapeUtil::ElementsIn(materialized->shape());
  }
  if (reduce.opcode() == HloOpcode::kReduce) {
    const int64_t minor_dim = input.has_layout()
                                  ? LayoutUtil::Minor(input.layout(), 0)
                                  : input.dimensions().size() - 1;
    work.reduces_minor_dim =
        absl::c_linear_search(reduce.dimensions(), minor_dim);
  }
  return work;
}

CpuPerformanceModel::ReduceRunTimes CpuPerformanceModel::EstimateReduce(
    const ReduceWork& work) const {
  const int64_t threads = std::max<int64_t>(1, device_info_.core_count());
  // Loop fusion: partitioned across outputs only; reading strided over a
  // major dimension forfeits vectorization.
  const int64_t loop_threads =
      std::min(threads, std::max<int64_t>(1, work.outputs));
  const int64_t loop_flops =
      work.reduces_minor_dim ? work.flops : work.flops * kStridedReducePenalty;
  ReduceRunTimes times;
  times.loop_fusion =
      CombineComputeAndMemoryAccessTime(
          ComputeTime(device_info_, loop_flops, work.type, loop_threads),
          ReadTimeWithDRAMHeuristic(device_info_, work.leaf_bytes,
                                    work.leaf_bytes)) +
      kKernelLaunchTime +
      (loop_threads > 1 ? kParallelTaskTime : absl::ZeroDuration());
  // Library fusion: a separate kernel writes the materialized input, the
  // library reads it and computes the rest on all threads.
  absl::Duration materialize = absl::ZeroDuration();
  if (work.materialized_bytes > 0) {
    materialize =
        CombineComputeAndMemoryAccessTime(
            ComputeTime(device_info_, work.materialized_flops, work.type,
                        threads),
            WriteTime(device_info_, work.materialized_bytes)) +
        kKernelLaunchTime + kParallelTaskTime;
  }
  const int64_t library_flops =
      std::max<int64_t>(0, work.flops - work.materialized_flops);
  times.library =
      materialize +
      CombineComputeAndMemoryAccessTime(
          ComputeTime(device_info_, library_flops, work.type, threads),
          ReadTimeWithDRAMHeuristic(
              device_info_, work.leaf_bytes + work.materialized_bytes,
              work.leaf_bytes + work.materialized_bytes)) +
      kLibraryCallTime;
  return times;
}

}  // namespace xla::cpu
