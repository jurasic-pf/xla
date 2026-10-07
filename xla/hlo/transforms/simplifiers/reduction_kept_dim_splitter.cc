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
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Returns the opcode of `reducer` if it applies one binary elementwise op to
// its two parameters, e.g. add(p0, p1).
std::optional<HloOpcode> SimpleReducerOpcode(const HloComputation* reducer) {
  const HloInstruction* root = reducer->root_instruction();
  if (reducer->num_parameters() != 2 || reducer->instruction_count() != 3 ||
      !root->IsElementwiseBinary()) {
    return std::nullopt;
  }
  const HloInstruction* lhs = root->operand(0);
  const HloInstruction* rhs = root->operand(1);
  if (lhs->opcode() != HloOpcode::kParameter ||
      rhs->opcode() != HloOpcode::kParameter ||
      lhs->parameter_number() != 0 || rhs->parameter_number() != 1) {
    return std::nullopt;
  }
  return root->opcode();
}

// Builds a reducer with `k` accumulators that applies `opcode` to each pair.
HloComputation* MakeVariadicReducer(HloModule* module, HloOpcode opcode,
                                    const Shape& scalar, int64_t k) {
  HloComputation::Builder builder("split_reducer");
  std::vector<HloInstruction*> params;
  for (int64_t i = 0; i < 2 * k; ++i) {
    params.push_back(builder.AddInstruction(
        HloInstruction::CreateParameter(i, scalar, "p")));
  }
  std::vector<HloInstruction*> results;
  for (int64_t i = 0; i < k; ++i) {
    results.push_back(builder.AddInstruction(HloInstruction::CreateBinary(
        scalar, opcode, params[i], params[k + i])));
  }
  builder.AddInstruction(HloInstruction::CreateTuple(results));
  return module->AddEmbeddedComputation(builder.Build());
}

}  // namespace

absl::StatusOr<bool> ReductionKeptDimSplitter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* reduce : computation->MakeInstructionPostOrder()) {
      if (reduce->opcode() != HloOpcode::kReduce ||
          reduce->operand_count() != 2 || !reduce->shape().IsArray()) {
        continue;
      }
      HloInstruction* input = reduce->mutable_operand(0);
      HloInstruction* init = reduce->mutable_operand(1);
      const Shape& input_shape = input->shape();
      const int64_t rank = input_shape.dimensions().size();
      if (rank < 2 || reduce->dimensions().empty()) {
        continue;
      }
      const int64_t last = rank - 1;
      const int64_t k = input_shape.dimensions(last);
      if (absl::c_linear_search(reduce->dimensions(), last) || k < 2 ||
          k > max_kept_dim_size_) {
        continue;
      }
      int64_t reduced_elements = 1;
      for (int64_t dim : reduce->dimensions()) {
        reduced_elements *= input_shape.dimensions(dim);
      }
      if (reduced_elements < min_reduced_elements_) {
        continue;
      }
      std::optional<HloOpcode> opcode =
          SimpleReducerOpcode(reduce->to_apply());
      if (!opcode.has_value()) {
        continue;
      }

      // Slices of the kept dimension, without that dimension.
      Shape slice_shape = input_shape;
      slice_shape.set_dimensions(last, 1);
      Shape slice_shape_squeezed =
          ShapeUtil::DeleteDimension(last, input_shape);
      std::vector<int64_t> start(rank, 0);
      std::vector<int64_t> limit(input_shape.dimensions().begin(),
                                 input_shape.dimensions().end());
      std::vector<int64_t> strides(rank, 1);
      std::vector<HloInstruction*> slices;
      for (int64_t i = 0; i < k; ++i) {
        start[last] = i;
        limit[last] = i + 1;
        HloInstruction* slice = computation->AddInstruction(
            HloInstruction::CreateSlice(slice_shape, input, start, limit,
                                        strides));
        slices.push_back(computation->AddInstruction(
            HloInstruction::CreateReshape(slice_shape_squeezed, slice)));
      }

      const Shape& out_shape = reduce->shape();
      const int64_t out_last = out_shape.dimensions().size() - 1;
      Shape part_shape = ShapeUtil::DeleteDimension(out_last, out_shape);
      std::vector<Shape> part_shapes(k, part_shape);
      std::vector<HloInstruction*> inits(k, init);
      HloComputation* reducer = MakeVariadicReducer(
          module, *opcode,
          ShapeUtil::MakeScalarShape(out_shape.element_type()), k);
      HloInstruction* split = computation->AddInstruction(
          HloInstruction::CreateReduce(ShapeUtil::MakeTupleShape(part_shapes),
                                       slices, inits, reduce->dimensions(),
                                       reducer));

      Shape unsqueezed = out_shape;
      unsqueezed.set_dimensions(out_last, 1);
      std::vector<HloInstruction*> parts;
      for (int64_t i = 0; i < k; ++i) {
        HloInstruction* gte = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(part_shape, split, i));
        parts.push_back(computation->AddInstruction(
            HloInstruction::CreateReshape(unsqueezed, gte)));
      }
      HloInstruction* concat = computation->AddInstruction(
          HloInstruction::CreateConcatenate(out_shape, parts, out_last));
      ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(reduce, concat));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
