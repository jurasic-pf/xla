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

#include "xla/service/cpu/cpu_hlo_cost_analysis.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class CpuHloCostAnalysisTest : public HloHardwareIndependentTestBase {
 public:
  CpuHloCostAnalysisTest() { options_.count_multiple_input_accesses = true; }

 protected:
  HloCostAnalysis::Options options_;
};

TEST_F(CpuHloCostAnalysisTest, ElementwiseFlops) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY e {
  p = f32[100] parameter(0)
  add = f32[100] add(p, p)
  divide = f32[100] divide(p, p)
  exp = f32[100] exponential(p)
  ROOT t = (f32[100], f32[100], f32[100]) tuple(add, divide, exp)
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuHloCostAnalysis analysis(options_);
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(*FindInstruction(module.get(), "add")), 100);
  EXPECT_EQ(analysis.flop_count(*FindInstruction(module.get(), "divide")),
            300);
  EXPECT_EQ(analysis.flop_count(*FindInstruction(module.get(), "exp")), 1100);
  EXPECT_EQ(analysis.transcendental_flop_count(
                *FindInstruction(module.get(), "exp")),
            1100);
  EXPECT_EQ(analysis.transcendental_flop_count(
                *FindInstruction(module.get(), "divide")),
            0);
}

TEST_F(CpuHloCostAnalysisTest, MatrixVectorDotReusesTheVector) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY e {
  m = f32[4096,128] parameter(0)
  v = f32[128] parameter(1)
  ROOT d = f32[4096] dot(m, v), lhs_contracting_dims={1},
      rhs_contracting_dims={0}
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuHloCostAnalysis analysis(options_);
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  EXPECT_FLOAT_EQ(analysis.operand_utilization(*dot, 0), 1);
  EXPECT_FLOAT_EQ(analysis.operand_utilization(*dot, 1), 4096);
}

TEST_F(CpuHloCostAnalysisTest, DeinterleavingSlicesCostTime) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY e {
  q = f32[1024,4] parameter(0)
  c = f32[1024,1] slice(q), slice={[0:1024], [1:2]}
  r = f32[512,4] slice(q), slice={[0:512], [0:4]}
  d = f64[1024,4] parameter(1)
  e = f64[1024,1] slice(d), slice={[0:1024], [1:2]}
  ROOT t = (f32[1024,1], f32[512,4], f64[1024,1]) tuple(c, r, e)
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuHloCostAnalysis analysis(options_);
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));
  EXPECT_EQ(analysis.flop_count(*FindInstruction(module.get(), "c")), 2048);
  EXPECT_EQ(analysis.flop_count(*FindInstruction(module.get(), "r")), 0);
  EXPECT_EQ(analysis.flop_count(*FindInstruction(module.get(), "e")), 0);
}

TEST_F(CpuHloCostAnalysisTest, F64LibmCallsAreMoreExpensive) {
  EXPECT_GT(
      CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(F64, HloOpcode::kLog),
      CpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(F32,
                                                          HloOpcode::kLog));
}

TEST_F(CpuHloCostAnalysisTest, FusionUtilizationThroughBroadcast) {
  absl::string_view hlo_string = R"(
HloModule m

fused {
  p0 = f32[1000] parameter(0)
  b = f32[1000,64] broadcast(p0), dimensions={0}
  ROOT exp = f32[1000,64] exponential(b)
}

ENTRY e {
  p = f32[1000] parameter(0)
  ROOT fusion = f32[1000,64] fusion(p), kind=kLoop, calls=fused
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuHloCostAnalysis analysis(options_);
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));

  const HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_FLOAT_EQ(analysis.operand_utilization(*fusion, 0), 64);
  EXPECT_EQ(analysis.operand_bytes_accessed(*fusion, 0), 64 * 1000 * 4);
  EXPECT_EQ(analysis.flop_count(*fusion), 11 * 64 * 1000);
}

TEST_F(CpuHloCostAnalysisTest, FusionFlopsScaleWithUtilization) {
  // The exponential is computed once per element of the broadcast.
  absl::string_view hlo_string = R"(
HloModule m

fused {
  p0 = f32[1000] parameter(0)
  exp = f32[1000] exponential(p0)
  ROOT b = f32[1000,64] broadcast(exp), dimensions={0}
}

ENTRY e {
  p = f32[1000] parameter(0)
  ROOT fusion = f32[1000,64] fusion(p), kind=kLoop, calls=fused
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuHloCostAnalysis analysis(options_);
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));

  const HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.flop_count(*fusion), 11 * 64 * 1000);
}

TEST_F(CpuHloCostAnalysisTest, SmallMinorBroadcastDoesNotRecompute) {
  // The loop over the minor dimension of size 3 is unrolled, so the
  // exponential is computed once per element of its operand.
  absl::string_view hlo_string = R"(
HloModule m

fused {
  p0 = f32[1000] parameter(0)
  exp = f32[1000] exponential(p0)
  ROOT b = f32[1000,3] broadcast(exp), dimensions={0}
}

ENTRY e {
  p = f32[1000] parameter(0)
  ROOT fusion = f32[1000,3] fusion(p), kind=kLoop, calls=fused
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CpuHloCostAnalysis analysis(options_);
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis));

  const HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_FLOAT_EQ(analysis.operand_utilization(*fusion, 0), 1);
  EXPECT_EQ(analysis.flop_count(*fusion), 11 * 1000);
}

}  // namespace
}  // namespace xla::cpu
