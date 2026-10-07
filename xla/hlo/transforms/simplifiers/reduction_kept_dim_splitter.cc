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
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
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
#include "xla/xla_data.pb.h"

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

// Returns index `k` of dimension `dim` of an instruction (keeping the dimension
// with size 1), pushing the slice up through the ops that produce it so that
// each index becomes its own chain. Values that do not depend on `dim`, e.g.
// the operand of a broadcast along `dim`, stay shared.
class SliceHoister {
 public:
  explicit SliceHoister(HloComputation* computation)
      : computation_(computation) {}

  HloInstruction* SliceAt(HloInstruction* x, int64_t dim, int64_t k) {
    auto key = std::make_tuple(x, dim, k);
    if (auto it = memo_.find(key); it != memo_.end()) {
      return it->second;
    }
    HloInstruction* result = Hoist(x, dim, k);
    memo_[key] = result;
    return result;
  }

 private:
  static Shape Unit(const Shape& shape, int64_t dim) {
    Shape out = shape;
    out.set_dimensions(dim, 1);
    return out;
  }

  HloInstruction* Add(std::unique_ptr<HloInstruction> instr) {
    return computation_->AddInstruction(std::move(instr));
  }

  HloInstruction* ExplicitSlice(HloInstruction* x, int64_t dim, int64_t k) {
    const int64_t rank = x->shape().dimensions().size();
    std::vector<int64_t> start(rank, 0);
    std::vector<int64_t> limit(x->shape().dimensions().begin(),
                               x->shape().dimensions().end());
    std::vector<int64_t> strides(rank, 1);
    start[dim] = k;
    limit[dim] = k + 1;
    return Add(HloInstruction::CreateSlice(Unit(x->shape(), dim), x, start,
                                           limit, strides));
  }

  HloInstruction* Hoist(HloInstruction* x, int64_t dim, int64_t k) {
    const Shape unit = Unit(x->shape(), dim);
    if (x->IsElementwise() && x->opcode() != HloOpcode::kConstant &&
        x->operand_count() > 0) {
      std::vector<HloInstruction*> operands;
      for (HloInstruction* operand : x->operands()) {
        operands.push_back(SliceAt(operand, dim, k));
      }
      Shape shape = unit;
      shape.set_element_type(x->shape().element_type());
      return Add(x->CloneWithNewOperands(shape, operands));
    }
    switch (x->opcode()) {
      case HloOpcode::kBroadcast: {
        const auto& dims = x->dimensions();
        auto it = absl::c_find(dims, dim);
        HloInstruction* operand = x->mutable_operand(0);
        if (it != dims.end()) {
          operand = SliceAt(operand, it - dims.begin(), k);
        }
        return Add(HloInstruction::CreateBroadcast(unit, operand, dims));
      }
      case HloOpcode::kConcatenate: {
        if (x->concatenate_dimension() == dim) {
          int64_t offset = 0;
          for (HloInstruction* operand : x->operands()) {
            int64_t size = operand->shape().dimensions(dim);
            if (k < offset + size) {
              return SliceAt(operand, dim, k - offset);
            }
            offset += size;
          }
          break;
        }
        std::vector<HloInstruction*> operands;
        for (HloInstruction* operand : x->operands()) {
          operands.push_back(SliceAt(operand, dim, k));
        }
        return Add(x->CloneWithNewOperands(unit, operands));
      }
      case HloOpcode::kSlice: {
        if (x->slice_strides(dim) <= 0) {
          break;
        }
        int64_t index = x->slice_starts(dim) + k * x->slice_strides(dim);
        HloInstruction* operand = x->mutable_operand(0);
        // Slice the other dimensions as before, and this one at `index`.
        std::vector<int64_t> start(x->slice_starts().begin(),
                                   x->slice_starts().end());
        std::vector<int64_t> limit(x->slice_limits().begin(),
                                   x->slice_limits().end());
        std::vector<int64_t> strides(x->slice_strides().begin(),
                                     x->slice_strides().end());
        HloInstruction* sliced = SliceAt(operand, dim, index);
        start[dim] = 0;
        limit[dim] = 1;
        strides[dim] = 1;
        bool identity = true;
        for (int64_t i = 0; i < unit.dimensions().size(); ++i) {
          identity &= start[i] == 0 && strides[i] == 1 &&
                      limit[i] == sliced->shape().dimensions(i);
        }
        if (identity) {
          return sliced;
        }
        return Add(HloInstruction::CreateSlice(unit, sliced, start, limit,
                                               strides));
      }
      case HloOpcode::kPad: {
        const PaddingConfig::PaddingConfigDimension& pad =
            x->padding_config().dimensions(dim);
        if (pad.interior_padding() != 0 || pad.edge_padding_low() < 0 ||
            pad.edge_padding_high() < 0) {
          break;
        }
        HloInstruction* operand = x->mutable_operand(0);
        int64_t index = k - pad.edge_padding_low();
        PaddingConfig config = x->padding_config();
        config.mutable_dimensions(dim)->set_edge_padding_low(0);
        config.mutable_dimensions(dim)->set_edge_padding_high(0);
        if (index < 0 || index >= operand->shape().dimensions(dim)) {
          // Only padding: a broadcast of the padding value.
          return Add(HloInstruction::CreateBroadcast(
              unit, x->mutable_operand(1), {}));
        }
        return Add(HloInstruction::CreatePad(unit, SliceAt(operand, dim, index),
                                             x->mutable_operand(1), config));
      }
      case HloOpcode::kTranspose: {
        int64_t operand_dim = x->dimensions(dim);
        HloInstruction* operand = SliceAt(x->mutable_operand(0), operand_dim, k);
        return Add(HloInstruction::CreateTranspose(unit, operand,
                                                   x->dimensions()));
      }
      case HloOpcode::kReduce: {
        if (x->operand_count() != 2 || !x->shape().IsArray()) {
          break;
        }
        // Output dimension `dim` is the `dim`-th kept input dimension.
        int64_t kept = -1;
        int64_t input_dim = 0;
        for (; input_dim < x->operand(0)->shape().dimensions().size();
             ++input_dim) {
          if (!absl::c_linear_search(x->dimensions(), input_dim) &&
              ++kept == dim) {
            break;
          }
        }
        HloInstruction* input = SliceAt(x->mutable_operand(0), input_dim, k);
        return Add(x->CloneWithNewOperands(unit, {input, x->mutable_operand(1)}));
      }
      default:
        break;
    }
    return ExplicitSlice(x, dim, k);
  }

  HloComputation* computation_;
  absl::flat_hash_map<std::tuple<HloInstruction*, int64_t, int64_t>,
                      HloInstruction*>
      memo_;
};

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
      Shape slice_shape_squeezed =
          ShapeUtil::DeleteDimension(last, input_shape);
      std::vector<HloInstruction*> slices;
      SliceHoister hoister(computation);
      for (int64_t i = 0; i < k; ++i) {
        slices.push_back(computation->AddInstruction(
            HloInstruction::CreateReshape(slice_shape_squeezed,
                                          hoister.SliceAt(input, last, i))));
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
