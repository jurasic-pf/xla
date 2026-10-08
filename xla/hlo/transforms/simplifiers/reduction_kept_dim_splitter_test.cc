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

#include "xla/hlo/transforms/simplifiers/reduction_kept_dim_splitter.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class ReductionKeptDimSplitterTest : public HloHardwareIndependentTestBase {
 protected:
  int64_t CountReduces(const HloModule& module, int64_t operand_rank) {
    int64_t count = 0;
    for (const HloInstruction* instr :
         module.entry_computation()->instructions()) {
      if (instr->opcode() == HloOpcode::kReduce &&
          instr->operand(0)->shape().dimensions().size() == operand_rank) {
        ++count;
      }
    }
    return count;
  }

  bool HasShape(const HloModule& module, absl::string_view shape) {
    return absl::c_any_of(
        module.entry_computation()->instructions(),
        [&](const HloInstruction* instr) {
          return ShapeUtil::HumanString(instr->shape()) == shape;
        });
  }
};

constexpr absl::string_view kAdd = R"(
add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}
)";

TEST_F(ReductionKeptDimSplitterTest, SplitsReductionKeepingSmallLastDim) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  x = f32[64,2048,3] parameter(0)
  c = f32[] constant(0)
  ROOT r = f32[64,3] reduce(x, c), dimensions={1}, to_apply=add
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConcatenate);
  EXPECT_EQ(CountReduces(*module, /*operand_rank=*/3), 0);
  EXPECT_EQ(CountReduces(*module, /*operand_rank=*/2), 1);
}

TEST_F(ReductionKeptDimSplitterTest, ComponentsAreComputedWithoutTheXyzArray) {
  // d = y_j - x_i is [64,2048,3]; after the split each component is computed
  // from slices of x and y, so no [64,128,3] value remains.
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  x = f32[64,3] parameter(0)
  y = f32[2048,3] parameter(1)
  bx = f32[64,2048,3] broadcast(x), dimensions={0,2}
  by = f32[64,2048,3] broadcast(y), dimensions={1,2}
  d = f32[64,2048,3] subtract(by, bx)
  dd = f32[64,2048,3] multiply(d, d)
  c = f32[] constant(0)
  ROOT r = f32[64,3] reduce(dd, c), dimensions={1}, to_apply=add
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  RunHloPass(HloDCE(), module.get()).IgnoreError();
  EXPECT_FALSE(HasShape(*module, "f32[64,2048,3]"));
  EXPECT_TRUE(HasShape(*module, "f32[64,2048]"));
}

TEST_F(ReductionKeptDimSplitterTest, KeepsSmallReductions) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  x = f32[4,8,3] parameter(0)
  c = f32[] constant(0)
  ROOT r = f32[4,3] reduce(x, c), dimensions={1}, to_apply=add
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReductionKeptDimSplitterTest, UnrollsReductionOverXyz) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  x = f32[64,128,3] parameter(0)
  xx = f32[64,128,3] multiply(x, x)
  c = f32[] constant(0)
  ROOT r = f32[64,128] reduce(xx, c), dimensions={2}, to_apply=add
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountReduces(*module, /*operand_rank=*/3), 0);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

TEST_F(ReductionKeptDimSplitterTest, MergesIndependentSiblingReductions) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  p = f32[64,128] parameter(0)
  q = f32[64,128] parameter(1)
  e = f32[64,128] exponential(p)
  x = f32[64,128] multiply(e, q)
  y = f32[64,128] add(e, q)
  c = f32[] constant(0)
  c1 = f32[] constant(0)
  a = f32[64] reduce(x, c), dimensions={1}, to_apply=add
  b = f32[64] reduce(y, c1), dimensions={1}, to_apply=add
  ROOT t = (f32[64], f32[64]) tuple(a, b)
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/false,
                                /*merge_sibling_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountReduces(*module, /*operand_rank=*/2), 1);
}

TEST_F(ReductionKeptDimSplitterTest, MergesSiblingsOfEachDependentStage) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  p = f32[64,128] parameter(0)
  q = f32[64,128] parameter(1)
  c = f32[] constant(0)
  e1 = f32[64,128] exponential(p)
  x1 = f32[64,128] multiply(e1, q)
  y1 = f32[64,128] add(e1, q)
  a1 = f32[64] reduce(x1, c), dimensions={1}, to_apply=add
  b1 = f32[64] reduce(y1, c), dimensions={1}, to_apply=add
  s = f32[64,128] broadcast(a1), dimensions={0}
  e2 = f32[64,128] exponential(s)
  x2 = f32[64,128] multiply(e2, q)
  y2 = f32[64,128] add(e2, q)
  a2 = f32[64] reduce(x2, c), dimensions={1}, to_apply=add
  b2 = f32[64] reduce(y2, c), dimensions={1}, to_apply=add
  ROOT t = (f32[64], f32[64], f32[64]) tuple(b1, a2, b2)
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/false,
                                /*merge_sibling_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountReduces(*module, /*operand_rank=*/2), 2);
}

TEST_F(ReductionKeptDimSplitterTest, DoesNotMergeReductionsToFewOutputs) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  p = f32[8,4096] parameter(0)
  q = f32[8,4096] parameter(1)
  e = f32[8,4096] exponential(p)
  x = f32[8,4096] multiply(e, q)
  y = f32[8,4096] add(e, q)
  c = f32[] constant(0)
  a = f32[8] reduce(x, c), dimensions={1}, to_apply=add
  b = f32[8] reduce(y, c), dimensions={1}, to_apply=add
  ROOT t = (f32[8], f32[8]) tuple(a, b)
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/false,
                                /*merge_sibling_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReductionKeptDimSplitterTest, DoesNotMergeReductionsWithoutSharedWork) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  x = f32[64,128] parameter(0)
  y = f32[64,128] parameter(1)
  c = f32[] constant(0)
  a = f32[64] reduce(x, c), dimensions={1}, to_apply=add
  b = f32[64] reduce(y, c), dimensions={1}, to_apply=add
  ROOT t = (f32[64], f32[64]) tuple(a, b)
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/false,
                                /*merge_sibling_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReductionKeptDimSplitterTest, DoesNotMergeMixedReducers) {
  // A (sum, max) reduction shares work with the sum but must not be rebuilt as
  // two sums.
  std::string hlo = R"(
HloModule m

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

sum_max {
  a0 = f32[] parameter(0)
  a1 = f32[] parameter(1)
  b0 = f32[] parameter(2)
  b1 = f32[] parameter(3)
  s = f32[] add(a0, b0)
  m = f32[] maximum(a1, b1)
  ROOT t = (f32[], f32[]) tuple(s, m)
}

ENTRY main {
  p = f32[64,2048] parameter(0)
  e = f32[64,2048] exponential(p)
  c = f32[] constant(0)
  a = f32[64] reduce(e, c), dimensions={1}, to_apply=add
  b = (f32[64], f32[64]) reduce(e, e, c, c), dimensions={1}, to_apply=sum_max
  b0 = f32[64] get-tuple-element(b), index=0
  ROOT t = (f32[64], f32[64]) tuple(a, b0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/false,
                                /*merge_sibling_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReductionKeptDimSplitterTest, SliceOfUnhoistableOpConverges) {
  // A slice of a dot cannot be hoisted; the pass must not keep re-creating it.
  std::string hlo = R"(
HloModule m

ENTRY main {
  x = f32[64,32] parameter(0)
  y = f32[32,2] parameter(1)
  d = f32[64,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT s = f32[64,1] slice(d), slice={[0:64], [1:2]}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/true,
                                /*merge_sibling_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ReductionKeptDimSplitterTest, DoesNotMergeDependentReductions) {
  std::string hlo = absl::StrCat("HloModule m\n", kAdd, R"(
ENTRY main {
  x = f32[64,128] parameter(0)
  c = f32[] constant(0)
  a = f32[64] reduce(x, c), dimensions={1}, to_apply=add
  ba = f32[64,128] broadcast(a), dimensions={0}
  y = f32[64,128] subtract(x, ba)
  ROOT b = f32[64] reduce(y, c), dimensions={1}, to_apply=add
}
)");
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ReductionKeptDimSplitter pass(/*max_kept_dim_size=*/8,
                                /*min_reduced_elements=*/1024,
                                /*unroll_small_reductions=*/false,
                                /*merge_sibling_reductions=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
