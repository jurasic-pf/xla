#include "xla/service/cpu/cpu_performance_model.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
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


CpuPerformanceModel::ReduceWork LargeWork(int64_t outputs) {
  CpuPerformanceModel::ReduceWork work;
  work.input_elements = int64_t{1} << 24;
  work.outputs = outputs;
  work.type = F64;
  // Two transcendentals and an add per input element.
  work.flops = 31 * work.input_elements;
  work.leaf_bytes = 8 * work.input_elements;
  work.materialized_bytes = 8 * work.input_elements;
  work.materialized_flops = 30 * work.input_elements;
  return work;
}

TEST(CpuPerformanceModelReduceTest, LibraryIsFasterForExpensiveFullReduction) {
  // A loop fusion computes a reduction to one output on one thread.
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  if (model.device_info().core_count() < 4) {
    GTEST_SKIP() << "Needs a multi-core host.";
  }
  CpuPerformanceModel::ReduceRunTimes times =
      model.EstimateReduce(LargeWork(/*outputs=*/1));
  EXPECT_LT(times.library, times.loop_fusion);
}

TEST(CpuPerformanceModelReduceTest, LoopFusionIsFasterForManyOutputs) {
  // With enough outputs, both use all threads, and the library fusion also
  // writes and reads the materialized input.
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  CpuPerformanceModel::ReduceRunTimes times =
      model.EstimateReduce(LargeWork(/*outputs=*/1 << 16));
  EXPECT_LT(times.loop_fusion, times.library);
}

TEST(CpuPerformanceModelReduceTest, LoopFusionIsFasterForTinyReduction) {
  // A tiny reduction is dominated by the cost of calling the library.
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  CpuPerformanceModel::ReduceWork work;
  work.input_elements = 3 * 64;
  work.outputs = 3;
  work.type = F64;
  work.flops = 10 * work.input_elements;
  work.leaf_bytes = 8 * work.input_elements;
  work.materialized_bytes = 8 * work.input_elements;
  work.materialized_flops = 9 * work.input_elements;
  CpuPerformanceModel::ReduceRunTimes times = model.EstimateReduce(work);
  EXPECT_LT(times.loop_fusion, times.library);
}

TEST(CpuPerformanceModelReduceTest, ColumnReductionIsSlowerInLoopFusion) {
  CpuPerformanceModel model(CpuPerformanceModel::DefaultDeviceInfo());
  CpuPerformanceModel::ReduceWork rows = LargeWork(/*outputs=*/1 << 12);
  CpuPerformanceModel::ReduceWork columns = rows;
  columns.reduces_minor_dim = false;
  EXPECT_LT(model.EstimateReduce(rows).loop_fusion,
            model.EstimateReduce(columns).loop_fusion);
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
  // sine (15) + multiply (1) + the reducer's add (1) per element.
  EXPECT_EQ(work.flops, 17 * 64 * 1024);
  EXPECT_EQ(work.leaf_bytes, 4 * 64 * 1024);
  EXPECT_EQ(work.materialized_bytes, 4 * 64 * 1024);
  EXPECT_EQ(work.materialized_flops, 15 * 64 * 1024);
  EXPECT_TRUE(work.reduces_minor_dim);
}

}  // namespace
}  // namespace xla::cpu
