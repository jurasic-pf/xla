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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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

TEST_F(CpuPerformanceModelTest, FusingIntoColumnReductionsAvoidsStridedReads) {
  // Both users reduce the producer; one reads it along dimension 0 with a
  // stride of 8 KiB. Recomputing the producer in both is faster.
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a, b)
}

ENTRY e {
  x = f32[2048] parameter(0)
  y = f32[2048] parameter(1)
  bx = f32[2048,2048] broadcast(x), dimensions={0}
  by = f32[2048,2048] broadcast(y), dimensions={1}
  d = f32[2048,2048] subtract(bx, by)
  s = f32[2048,2048] sqrt(d)
  c = f32[] constant(0)
  rows = f32[2048] reduce(s, c), dimensions={1}, to_apply=add
  cols = f32[2048] reduce(s, c), dimensions={0}, to_apply=add
  ROOT t = (f32[2048], f32[2048]) tuple(rows, cols)
})";
  CpuPerformanceModel::RunTimes run_times = EstimateRunTimes(hlo_string, "s");
  EXPECT_LT(run_times.time_fused, run_times.time_unfused);
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

TEST(CpuPerformanceModelHostTest, DefaultDeviceInfoCountsPhysicalCores) {
  se::DeviceDescription info = CpuPerformanceModel::DefaultDeviceInfo();
  LOG(INFO) << "cores: " << info.core_count()
            << ", threads per core: " << info.threads_per_core_limit();
  EXPECT_GE(info.core_count(), 1);
  EXPECT_GE(info.threads_per_core_limit(), 1);
}

TEST(CpuPerformanceModelHostTest, OnlyPageStridesLoadACacheLinePerElement) {
  Shape rows = ShapeUtil::MakeShapeWithDescendingLayout(F32, {4096, 4096});
  Shape narrow = ShapeUtil::MakeShapeWithDescendingLayout(F32, {4096, 64});
  EXPECT_EQ(CpuPerformanceModel::StridedReadFactor(rows, {1}), 1);
  EXPECT_EQ(CpuPerformanceModel::StridedReadFactor(rows, {0}), 16);
  EXPECT_EQ(CpuPerformanceModel::StridedReadFactor(narrow, {0}), 1);
}

TEST(CpuPerformanceModelHostTest, ParallelKernelsPayForkJoin) {
  EXPECT_EQ(CpuPerformanceModel::KernelOverhead(1),
            CpuPerformanceModel::kKernelLaunchTime);
  EXPECT_GT(CpuPerformanceModel::KernelOverhead(16),
            CpuPerformanceModel::KernelOverhead(4));
}

TEST_F(CpuPerformanceModelTest, EntryTempCountsThePeakOfLiveValues) {
  // `big` dies before `small` is computed, so they share memory and
  // materializing `small` does not grow the temp buffer.
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY e {
  p = f32[8388608] parameter(0)
  q = f32[1024] parameter(1)
  z = f32[] constant(0)
  big = f32[8388608] exponential(p)
  r1 = f32[] reduce(big, z), dimensions={0}, to_apply=add
  r2 = f32[] reduce(big, z), dimensions={0}, to_apply=add
  small = f32[1024] exponential(q)
  m1 = f32[1024] multiply(small, q)
  m2 = f32[1024] add(small, q)
  ROOT t = (f32[], f32[], f32[1024], f32[1024]) tuple(r1, r2, m1, m2)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  const HloComputation& entry = *module->entry_computation();
  CpuPerformanceModel::EntryTemp temp(
      entry,
      [](const HloInstruction& instr) {
        return instr.opcode() != HloOpcode::kParameter &&
               instr.opcode() != HloOpcode::kConstant &&
               instr.shape().IsArray() &&
               (instr.user_count() > 1 ||
                instr.users().front()->opcode() == HloOpcode::kTuple);
      },
      [](const HloInstruction& instr) {
        return instr.opcode() != HloOpcode::kTuple;
      });
  constexpr int64_t kBigBytes = int64_t{32} << 20;
  // `big` and the two scalar reductions of it.
  EXPECT_EQ(temp.peak_bytes(), kBigBytes + 8);
  EXPECT_EQ(temp.PeakIncrease(*FindInstruction(module.get(), "small"), 4096),
            0);
  // Without `big`, only the small values are live.
  const int64_t big_increase =
      temp.PeakIncrease(*FindInstruction(module.get(), "big"), kBigBytes);
  EXPECT_GT(big_increase, kBigBytes - 4 * 4096);
  EXPECT_LE(big_increase, kBigBytes);
  // A value that is not counted adds its bytes where it is live.
  EXPECT_EQ(temp.PeakIncrease(*FindInstruction(module.get(), "r1"), 64), 64);
  // Once `big` is materialized, a value of its size live at the same time
  // doubles the peak.
  temp.Materialize(*FindInstruction(module.get(), "big"));
  EXPECT_EQ(temp.PeakIncrease(*FindInstruction(module.get(), "r1"), kBigBytes),
            kBigBytes);
}

TEST_F(CpuPerformanceModelTest, EntryTempValueIsLiveUntilTheKernelsOfItsUsers) {
  // `n` is not materialized, so it is computed in the kernel of `m`, and a
  // materialized `g` is read there, while `big` is live.
  absl::string_view hlo_string = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY e {
  p = f32[8388608] parameter(0)
  z = f32[] constant(0)
  g = f32[8388608] exponential(p)
  n = f32[8388608] negate(g)
  big = f32[8388608] sine(p)
  r = f32[] reduce(big, z), dimensions={0}, to_apply=add
  b = f32[8388608] broadcast(r), dimensions={}
  ROOT m = f32[8388608] add(n, b)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CpuPerformanceModel::EntryTemp temp(
      *module->entry_computation(),
      [](const HloInstruction& instr) { return instr.name() == "big"; },
      [](const HloInstruction& instr) { return true; });
  constexpr int64_t kBigBytes = int64_t{32} << 20;
  EXPECT_EQ(temp.peak_bytes(), kBigBytes);
  const HloInstruction& g = *FindInstruction(module.get(), "g");
  EXPECT_EQ(temp.PeakIncrease(g, kBigBytes), kBigBytes);
  // Once `n` is materialized, `g` is read only by its kernel.
  temp.Materialize(*FindInstruction(module.get(), "n"));
  EXPECT_EQ(temp.peak_bytes(), 2 * kBigBytes);
  EXPECT_EQ(temp.PeakIncrease(g, kBigBytes), 0);
}

}  // namespace
}  // namespace xla::cpu
