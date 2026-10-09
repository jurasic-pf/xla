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
  // The C library returns freed allocations larger than its maximum mmap
  // threshold to the system, so a temp buffer that large is faulted in again
  // by every execution, at this cost per page.
  static constexpr int64_t kFreshAllocationBytes = int64_t{32} << 20;
  static constexpr int64_t kPageFaultBytes = 4096;
  static constexpr absl::Duration kPageFaultTime = absl::Nanoseconds(900);
  // Returns the cost of the page faults of materializing `bytes` once per
  // execution.
  static absl::Duration FreshAllocationTime(int64_t bytes);
  // Returns true if `instr` runs once per execution, not in a loop body.
  static bool RunsOncePerExecution(const HloInstruction& instr);
  // Whether the module's temp buffer is larger than kFreshAllocationBytes, so
  // every byte materialized once per execution is faulted in, and the cost of
  // these page faults for materializing `bytes` of `producer`.
  void set_temp_is_fresh(bool fresh) { temp_is_fresh_ = fresh; }
  absl::Duration MaterializeFaultTime(const HloInstruction& producer,
                                      int64_t bytes) const;

  // Bandwidth of one core streaming from memory, and from its L2 cache.
  static constexpr double kCoreMemoryBandwidth = 20e9;
  static constexpr double kCoreCacheBandwidth = 170e9;
  // A reduction streaming its input from memory reads less per core, measured
  // with loop and library reductions of 2^24 elements: 28 GB/s on 4 cores,
  // 40-51 GB/s on 7.
  static constexpr double kCoreReduceReadBandwidth = 7e9;
  static constexpr double kReduceReadBandwidth = 51e9;
  // Time for a reduction on `cores` cores to read `bytes`.
  absl::Duration ReduceReadTime(int64_t bytes, double cores) const;
  // A library reduction computes tiles of this many input bytes as parallel
  // tasks. If one output reduces more than a tile, it splits the reduced
  // dimensions into partial reductions, and needs this many tiles per thread
  // to balance them.
  static constexpr int64_t kLibraryReduceTileBytes = 128 << 10;
  static constexpr int64_t kLibraryPartialReduceTilesPerThread = 4;
  // A loop that reads an array along a dimension whose stride is at least a
  // page defeats the hardware prefetchers and loads a cache line per element.
  static constexpr int64_t kPageBytes = 4096;
  static constexpr int64_t kCacheLineBytes = 64;
  // Returns the bytes a loop fusion loads per byte of `input` when it reduces
  // `reduced_dims`, iterating over the innermost reduced dimension.
  static int64_t StridedReadFactor(const Shape& input,
                                   absl::Span<const int64_t> reduced_dims);

  // Returns how many times more bytes `consumer` loads to read `operand` than
  // the operand has: a reduction that reads its input along a major dimension
  // with a large stride loads a cache line per element.
  static int64_t ReadFactor(const HloInstruction* consumer,
                            const HloInstruction* operand);

  // A library reduction over the minor dimension computes other ops of its
  // input faster than a loop fusion, as it vectorizes along that dimension.
  // Measured with a 5-op chain from broadcasts, reduced in cache on one
  // thread: f32 0.048 vs 0.181 ns, f64 0.084 vs 0.256 ns per element. Keeping
  // a minor dimension of 3 or 8 elements, it has no such advantage (0.9-1.5x
  // at 16 threads).
  static constexpr double kLibraryArithmeticSpeedupF32 = 3.8;
  static constexpr double kLibraryArithmeticSpeedupF64 = 3.0;

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
  // Cores that run `kernel`, a loop fusion with the given work. The loop
  // emitter only partitions the kernel's output.
  int64_t KernelCores(const HloInstruction& kernel, int64_t flops,
                      int64_t transcendental_flops, int64_t bytes) const;
  // Scales `compute_time`, `*read_time` and `*write_time`, estimated for all
  // cores, to `cores`. Returns the compute time.
  absl::Duration ScaleToCores(absl::Duration compute_time, int64_t cores,
                              absl::Duration* read_time,
                              absl::Duration* write_time) const;
  // Hardware threads of the intra-op thread pool.
  int64_t Threads() const;

  // Work of a loop-fusible computation, per element unless noted.
  struct ChainWork {
    // Time of one element in a loop fusion and in a library fusion, which
    // vectorizes transcendentals, in units of an add. Fractional where a
    // value is shared by several elements.
    double flops = 0;
    double library_flops = 0;
    // Part of `flops` and `library_flops` spent in transcendental ops.
    double transcendental_flops = 0;
    double library_transcendental_flops = 0;
    // Total bytes of the chain's leaves, e.g. parameters, and of those that
    // are read along the loop rather than through broadcasts.
    int64_t leaf_bytes = 0;
    int64_t streamed_bytes = 0;
  };

  // Work of a reduction whose input is computed by a chain of loop-fusible
  // instructions.
  struct ReduceWork {
    std::string ToString() const;

    int64_t input_elements = 0;
    int64_t outputs = 0;
    // Computing the reduction input from the chain's leaves and reducing it.
    ChainWork chain;
    // An input that a library fusion cannot absorb, which a loop fusion
    // writes and the library fusion reads: its bytes, and the work to
    // compute it. The library fusion computes the rest of the chain.
    int64_t materialized_bytes = 0;
    int64_t materialized_elements = 0;
    bool materialized_anyway = false;
    // Whether the reduction runs once per execution, not in a loop body.
    bool once_per_execution = false;
    ChainWork materialized;
    // Other inputs that the library fusion cannot absorb, which are written
    // for it as well: totals of their work and bytes.
    int64_t extra_materialized_flops = 0;
    int64_t extra_materialized_transcendental_flops = 0;
    int64_t extra_materialized_bytes = 0;
    int64_t extra_materialized_leaf_bytes = 0;
    // Bytes loaded per byte of streamed input, see StridedReadFactor.
    int64_t strided_read_factor = 1;
    // Whether the minor dimension of the input is reduced.
    bool reduces_minor_dim = true;
    // Whether a loop fusion can vectorize the reducer, i.e. reassociate it.
    bool loop_reducer_vectorizes = true;
    PrimitiveType type = F32;
  };

  // Returns the work of `reduce`, computing its input in a loop fusion. A
  // library fusion of `reduce` reads `materialized`, if not null. If it is
  // `materialized_anyway`, e.g. for other users, the loop fusion reads it as
  // well and neither fusion pays for writing it.
  static ReduceWork AnalyzeReduce(
      const HloInstruction& reduce, const HloInstruction* materialized,
      bool materialized_anyway = false,
      absl::Span<const HloInstruction* const> also_materialized = {});

  // Number of parallel tasks of `work` as a loop fusion, and whether the loop
  // fusion needs TreeReductionRewriter to split it into a reduce-window and a
  // small reduction to use them: the loop emitter only partitions outputs.
  int64_t LoopFusionTasks(const ReduceWork& work) const;
  bool LoopFusionSplitsReduce(const ReduceWork& work) const;

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
  bool temp_is_fresh_ = false;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_PERFORMANCE_MODEL_H_
