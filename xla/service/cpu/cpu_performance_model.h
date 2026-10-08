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

#ifndef XLA_SERVICE_CPU_CPU_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_CPU_CPU_PERFORMANCE_MODEL_H_

#include <cstdint>
#include <string>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

struct EstimateRunTimeData {
  int64_t flops;
  int64_t bytes_read;
  int64_t bytes_written;
  absl::Duration read_time;
  absl::Duration write_time;
  absl::Duration compute_time;
  absl::Duration exec_time;

  // Returns an estimate that is guaranteed to be larger than any real runtime.
  static EstimateRunTimeData Infinite();

  // Returns true if the estimate is guaranteed to be larger than any real
  // runtime.
  bool IsInfinite() const { return exec_time == absl::InfiniteDuration(); }

  std::string ToString() const;
};

// Analytical performance model of loop fusion on CPU, following
// GpuPerformanceModel: a fusion's run time combines the time to compute its
// flops and the time to read its operands and write its outputs.
class CpuPerformanceModel {
 public:
  struct RunTimes {
    absl::Duration time_unfused;
    absl::Duration time_fused;
  };

  // See GpuPerformanceModelBase::kMemoryComputeParallelism.
  static constexpr double kMemoryComputeParallelism = 0.95;

  // Parameters of the host, with throughputs measured on an AVX2 x86 host.
  // core_count counts physical cores: hyperthreads share a core's vector
  // units and add no throughput to vectorized loops.
  static se::DeviceDescription DefaultDeviceInfo();

  explicit CpuPerformanceModel(const se::DeviceDescription& device_info)
      : device_info_(device_info) {}

  const se::DeviceDescription& device_info() const { return device_info_; }

  EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr,
      const CpuHloCostAnalysis* cost_analysis) const;

  // Estimates the run time of `consumer` with `producer` fused into it.
  absl::Duration EstimateRunTimeForFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const CpuHloCostAnalysis* cost_analysis) const;

  // Estimates the run time of `producer` and `fused_consumers` with and without
  // fusing `producer` into each of `fused_consumers`.
  RunTimes EstimateRunTimes(
      const HloInstruction* producer, const CpuHloCostAnalysis* cost_analysis,
      absl::Span<const HloInstruction* const> fused_consumers) const;

  // Returns bytes accessed of operand output by instruction. Returns 0, if the
  // operand is not used by the instruction.
  static int64_t GetOperandBytesAccessed(
      const CpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
      const HloInstruction* operand);

  // Returns utilization of operand by instruction. Returns 0, if the operand is
  // not used by the instruction.
  static float GetOperandUtilization(const CpuHloCostAnalysis* cost_analysis,
                                     const HloInstruction* instr,
                                     const HloInstruction* operand);

  // Returns bytes accessed of operand after producer and consumer are fused
  // together.
  static int64_t GetSharedOperandBytesAccessed(
      const CpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
      const HloInstruction* consumer, const HloInstruction* operand);

  // Estimates the time to read `n_bytes_total` bytes of an operand of size
  // `n_bytes_net`. The first `n_bytes_net` bytes are read from memory, or from
  // the cache if they fit into it; the remaining bytes are read from the cache.
  static absl::Duration ReadTimeWithDRAMHeuristic(
      const se::DeviceDescription& device_info, int64_t n_bytes_net,
      int64_t n_bytes_total);

  static absl::Duration WriteTime(const se::DeviceDescription& device_info,
                                  int64_t bytes_written);

  static absl::Duration ComputeTime(const se::DeviceDescription& device_info,
                                    int64_t flops);

  static absl::Duration CombineComputeAndMemoryAccessTime(
      absl::Duration compute_time, absl::Duration memory_access_time);

  // Fixed costs, measured with microbenchmarks of single operations on an
  // 8-core, 16-thread AVX2 x86 host. Launching one kernel thunk:
  static constexpr absl::Duration kKernelLaunchTime = absl::Nanoseconds(20);
  // Calling a library (YNNPACK) fusion:
  static constexpr absl::Duration kLibraryCallTime = absl::Nanoseconds(150);
  // Running a kernel as `tasks` > 1 parallel tasks on the intra-op thread
  // pool and waiting for them: kForkJoinTime + tasks * kForkJoinTaskTime.
  static constexpr absl::Duration kForkJoinTime = absl::Nanoseconds(4700);
  static constexpr absl::Duration kForkJoinTaskTime = absl::Nanoseconds(1230);
  // Bandwidth of one core streaming from memory, and from its L2 cache.
  static constexpr double kCoreMemoryBandwidth = 20e9;
  static constexpr double kCoreCacheBandwidth = 170e9;
  // A library reduction computes tiles of this many input bytes as parallel
  // tasks. If one output reduces more than a tile, it splits the reduced
  // dimensions into partial reductions, and needs this many tiles per thread
  // to balance them.
  static constexpr int64_t kLibraryReduceTileBytes = 128 << 10;
  static constexpr int64_t kLibraryPartialReduceTilesPerThread = 4;
  // Fraction of the cores that a library fusion keeps busy.
  static constexpr double kLibraryParallelEfficiency = 0.85;

  // Time to compute `flops` of `type` on `threads` cores. Unlike
  // ComputeTime, counts the vector width of `type`.
  static absl::Duration ComputeTime(const se::DeviceDescription& device_info,
                                    int64_t flops, PrimitiveType type,
                                    int64_t threads);

  // Time to read or write `bytes` on `threads` cores, from their L2 caches if
  // the bytes fit into them.
  static absl::Duration MemoryTime(const se::DeviceDescription& device_info,
                                   int64_t bytes, int64_t threads);

  // Fixed cost of running `tasks` parallel tasks, and of a kernel that runs
  // as `tasks` parallel tasks.
  static absl::Duration ForkJoinTime(int64_t tasks);
  static absl::Duration KernelOverhead(int64_t tasks);

  // Number of parallel tasks of a loop fusion, following
  // ParallelTaskAssigner, and the cores that run them.
  int64_t LoopFusionTasks(int64_t flops, int64_t transcendental_flops,
                          int64_t bytes) const;
  int64_t Cores(int64_t tasks) const;
  // Hardware threads of the intra-op thread pool.
  int64_t Threads() const;

  // Work of a loop-fusible computation, per element unless noted.
  struct ChainWork {
    // Time of one element in a loop fusion and in a library fusion, which
    // vectorizes transcendentals, in units of an add.
    int64_t flops = 0;
    int64_t library_flops = 0;
    // Part of `flops` spent in transcendental ops.
    int64_t transcendental_flops = 0;
    // Total bytes of the chain's leaves, e.g. parameters.
    int64_t leaf_bytes = 0;
  };

  // Work of a reduction whose input is computed by a chain of loop-fusible
  // instructions.
  struct ReduceWork {
    int64_t input_elements = 0;
    int64_t outputs = 0;
    // Computing the reduction input from the chain's leaves and reducing it.
    ChainWork chain;
    // An input that a library fusion cannot absorb, which a loop fusion
    // writes and the library fusion reads: its bytes, and the work to
    // compute it. The library fusion computes the rest of the chain.
    int64_t materialized_bytes = 0;
    int64_t materialized_elements = 0;
    ChainWork materialized;
    // Whether the minor dimension of the input is reduced. Loop fusion reads
    // the input strided otherwise.
    bool reduces_minor_dim = true;
    PrimitiveType type = F32;
  };

  // Returns the work of `reduce`, computing its input in a loop fusion. A
  // library fusion of `reduce` reads `materialized`, if not null.
  static ReduceWork AnalyzeReduce(const HloInstruction& reduce,
                                  const HloInstruction* materialized);

  struct ReduceRunTimes {
    absl::Duration loop_fusion;
    absl::Duration library;
  };

  // Estimates `work` as a loop fusion and as a library fusion, which
  // vectorizes transcendentals and splits large reductions across threads,
  // but reads materialized inputs.
  ReduceRunTimes EstimateReduce(const ReduceWork& work) const;

 private:
  se::DeviceDescription device_info_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_PERFORMANCE_MODEL_H_
