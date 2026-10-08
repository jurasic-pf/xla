#include "xla/service/cpu/cpu_performance_model.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class CpuPerformanceModelTest : public HloHardwareIndependentTestBase {
 public:
  CpuPerformanceModelTest() { options_.count_multiple_input_accesses = true; }

  CpuPerformanceModel::RunTimes EstimateRunTimes(absl::string_view hlo_string,
                                                 absl::string_view producer) {
    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    CpuHloCostAnalysis analysis(options_);
    CHECK_OK(module->entry_computation()->Accept(&analysis));
    const HloInstruction* instr = FindInstruction(module.get(), producer);
    std::vector<const HloInstruction*> consumers(instr->users().begin(),
                                                 instr->users().end());
    return model_.EstimateRunTimes(instr, &analysis, consumers);
  }

 protected:
  HloCostAnalysis::Options options_;
  CpuPerformanceModel model_{CpuPerformanceModel::DefaultDeviceInfo()};
};

TEST_F(CpuPerformanceModelTest, FusingCheapProducerIntoBroadcastIsFaster) {
  // The producer is read three times per element. Recomputing a sqrt is
  // cheaper than writing and reading back 4 MiB.
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a, b)
}

fused {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024,3] parameter(1)
  b = f32[1024,1024,3] broadcast(p0), dimensions={0,1}
  d = f32[1024,1024,3] divide(p1, b)
  c = f32[] constant(0)
  ROOT r = f32[1024,3] reduce(d, c), dimensions={1}, to_apply=add
}

ENTRY e {
  x = f32[1024,1024] parameter(0)
  y = f32[1024,1024,3] parameter(1)
  sqrt = f32[1024,1024] sqrt(x)
  ROOT fusion = f32[1024,3] fusion(sqrt, y), kind=kLoop, calls=fused
})";
  CpuPerformanceModel::RunTimes run_times =
      EstimateRunTimes(hlo_string, "sqrt");
  EXPECT_LT(run_times.time_fused, run_times.time_unfused);
}

TEST_F(CpuPerformanceModelTest, FusingExpensiveProducerIntoHeavyReuseIsSlower) {
  // Fusing the exponential would recompute it 64 times per element.
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a, b)
}

fused {
  p0 = f32[131072] parameter(0)
  b = f32[131072,64] broadcast(p0), dimensions={0}
  c = f32[] constant(0)
  ROOT r = f32[64] reduce(b, c), dimensions={0}, to_apply=add
}

ENTRY e {
  x = f32[131072] parameter(0)
  exp = f32[131072] exponential(x)
  ROOT fusion = f32[64] fusion(exp), kind=kLoop, calls=fused
})";
  CpuPerformanceModel::RunTimes run_times = EstimateRunTimes(hlo_string, "exp");
  EXPECT_GT(run_times.time_fused, run_times.time_unfused);
}

TEST_F(CpuPerformanceModelTest, CustomCallIsNeverFused) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY e {
  x = f32[1024] parameter(0)
  cc = f32[1024] custom-call(x), custom_call_target="foo"
  ROOT n = f32[1024] negate(cc)
})";
  CpuPerformanceModel::RunTimes run_times = EstimateRunTimes(hlo_string, "cc");
  EXPECT_EQ(run_times.time_fused, absl::InfiniteDuration());
}


// sin(x) * cos(x) of f64[n], reduced to `outputs` outputs. The library fusion
// reads `materialized_ops` of sin and cos from memory.
CpuPerformanceModel::ReduceWork TrigWork(int64_t n, int64_t outputs,
                                         int materialized_ops) {
  const int64_t trig = CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(
      F64, HloOpcode::kSin);
  const int64_t vectorized =
      CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(F64,
                                                          HloOpcode::kExp);
  CpuPerformanceModel::ReduceWork work;
  work.input_elements = n;
  work.outputs = outputs;
  work.type = F64;
  work.chain.flops = 2 * trig + 2;
  work.chain.transcendental_flops = 2 * trig;
  work.chain.library_flops =
      (2 - materialized_ops) * vectorized + materialized_ops + 1;
  work.chain.leaf_bytes = 8 * n;
  if (materialized_ops > 0) {
    work.materialized_bytes = 8 * n;
    work.materialized_elements = n;
    work.materialized.flops = materialized_ops * trig;
    work.materialized.transcendental_flops = work.materialized.flops;
    work.materialized.leaf_bytes = 8 * n;
  }
  return work;
}

TEST(CpuPerformanceModelReduceTest, LibraryIsFasterIfItComputesTheChain) {
  // The library fusion vectorizes sin and cos.
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  CpuPerformanceModel::ReduceRunTimes times = model.EstimateReduce(
      TrigWork(int64_t{1} << 24, /*outputs=*/1, /*materialized_ops=*/0));
  EXPECT_LT(times.library, times.loop_fusion);
}

TEST(CpuPerformanceModelReduceTest, LoopFusionIsFasterIfSineIsMaterialized) {
  // Both compute sin with the same scalar code; the library fusion also
  // writes and reads it, and launches a second kernel.
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  CpuPerformanceModel::ReduceRunTimes times = model.EstimateReduce(
      TrigWork(int64_t{1} << 18, /*outputs=*/1, /*materialized_ops=*/2));
  EXPECT_LT(times.loop_fusion, times.library);
}

TEST(CpuPerformanceModelReduceTest, LoopFusionIsFasterForTinyReduction) {
  // A tiny reduction of cheap ops is dominated by the fixed costs of the
  // second kernel and the library call.
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  CpuPerformanceModel::ReduceWork work =
      TrigWork(3 * 64, /*outputs=*/3, /*materialized_ops=*/1);
  work.chain.flops = work.chain.library_flops = 3;
  work.chain.transcendental_flops = 0;
  work.materialized.flops = 1;
  work.materialized.transcendental_flops = 0;
  CpuPerformanceModel::ReduceRunTimes times = model.EstimateReduce(work);
  EXPECT_LT(times.loop_fusion, times.library);
}

TEST(CpuPerformanceModelReduceTest, ColumnReductionIsSlowerInLoopFusion) {
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  CpuPerformanceModel::ReduceWork rows =
      TrigWork(int64_t{1} << 24, /*outputs=*/1 << 12, /*materialized_ops=*/0);
  rows.chain.flops = 2;
  rows.chain.transcendental_flops = 0;
  CpuPerformanceModel::ReduceWork columns = rows;
  columns.reduces_minor_dim = false;
  EXPECT_LT(model.EstimateReduce(rows).loop_fusion,
            model.EstimateReduce(columns).loop_fusion);
}

TEST(CpuPerformanceModelReduceTest, DefaultDeviceInfoCountsPhysicalCores) {
  se::DeviceDescription info = CpuPerformanceModel::DefaultDeviceInfo();
  LOG(INFO) << "cores: " << info.core_count()
            << ", threads per core: " << info.threads_per_core_limit();
  EXPECT_GE(info.core_count(), 1);
  EXPECT_GE(info.threads_per_core_limit(), 1);
}

TEST(CpuPerformanceModelReduceTest, ParallelKernelsPayForkJoin) {
  EXPECT_EQ(CpuPerformanceModel::KernelOverhead(1),
            CpuPerformanceModel::kKernelLaunchTime);
  EXPECT_GT(CpuPerformanceModel::KernelOverhead(16),
            CpuPerformanceModel::KernelOverhead(4));
}

TEST_F(CpuPerformanceModelTest, AnalyzeReduceCountsTheRecomputedChain) {
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a, b)
}

ENTRY e {
  x = f32[64,1024] parameter(0)
  s = f32[64,1024] sine(x)
  m = f32[64,1024] multiply(s, s)
  c = f32[] constant(0)
  ROOT r = f32[64] reduce(m, c), dimensions={1}, to_apply=add
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  const HloInstruction* sine = reduce->operand(0)->operand(0);
  CpuPerformanceModel::ReduceWork work =
      CpuPerformanceModel::AnalyzeReduce(*reduce, /*materialized=*/sine);
  EXPECT_EQ(work.input_elements, 64 * 1024);
  EXPECT_EQ(work.outputs, 64);
  const int64_t sine_flops =
      CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(F32,
                                                          HloOpcode::kSin);
  // sine + multiply + the reducer's add per element.
  EXPECT_EQ(work.chain.flops, sine_flops + 2);
  EXPECT_EQ(work.chain.transcendental_flops, sine_flops);
  // The library fusion reads sine and computes the rest.
  EXPECT_EQ(work.chain.library_flops, 2);
  EXPECT_EQ(work.chain.leaf_bytes, 4 * 64 * 1024);
  EXPECT_EQ(work.materialized_bytes, 4 * 64 * 1024);
  EXPECT_EQ(work.materialized.flops, sine_flops);
  EXPECT_EQ(work.materialized.leaf_bytes, 4 * 64 * 1024);
  EXPECT_TRUE(work.reduces_minor_dim);
}

}  // namespace
}  // namespace xla::cpu
