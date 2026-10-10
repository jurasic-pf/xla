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
#include "absl/types/span.h"
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
      !root->IsElementwiseBinary() || root->opcode() == HloOpcode::kCompare) {
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

// Returns index `k` of dimension `dim` of an instruction, without that
// dimension, pushing the selection up through the ops that produce it so that
// each index becomes its own chain of rank-reduced ops. Values that do not
// depend on `dim`, e.g. the operand of a broadcast along `dim`, stay shared.
class ComponentHoister {
 public:
  explicit ComponentHoister(HloComputation* computation)
      : computation_(computation) {}

  HloInstruction* Component(HloInstruction* x, int64_t dim, int64_t k) {
    auto key = std::make_tuple(x, dim, k);
    if (auto it = memo_.find(key); it != memo_.end()) {
      return it->second;
    }
    HloInstruction* result = Hoist(x, dim, k);
    memo_[key] = result;
    return result;
  }

 private:
  // Drops `dim` from a list of dimension numbers, renumbering the rest.
  static std::vector<int64_t> DropDim(absl::Span<const int64_t> dims,
                                      int64_t dim) {
    std::vector<int64_t> out;
    for (int64_t d : dims) {
      if (d != dim) {
        out.push_back(d > dim ? d - 1 : d);
      }
    }
    return out;
  }

  HloInstruction* Add(std::unique_ptr<HloInstruction> instr) {
    return computation_->AddInstruction(std::move(instr));
  }

  HloInstruction* SliceAndSqueeze(HloInstruction* x, int64_t dim, int64_t k) {
    const int64_t rank = x->shape().dimensions().size();
    std::vector<int64_t> start(rank, 0);
    std::vector<int64_t> limit(x->shape().dimensions().begin(),
                               x->shape().dimensions().end());
    std::vector<int64_t> strides(rank, 1);
    start[dim] = k;
    limit[dim] = k + 1;
    Shape unit = x->shape();
    unit.set_dimensions(dim, 1);
    HloInstruction* slice = Add(
        HloInstruction::CreateSlice(unit, x, start, limit, strides));
    return Add(HloInstruction::CreateReshape(
        ShapeUtil::DeleteDimension(dim, x->shape()), slice));
  }

  HloInstruction* Hoist(HloInstruction* x, int64_t dim, int64_t k) {
    const Shape squeezed = ShapeUtil::DeleteDimension(dim, x->shape());
    if (x->IsElementwise() && x->opcode() != HloOpcode::kConstant &&
        x->operand_count() > 0) {
      std::vector<HloInstruction*> operands;
      for (HloInstruction* operand : x->operands()) {
        operands.push_back(Component(operand, dim, k));
      }
      return Add(x->CloneWithNewOperands(squeezed, operands));
    }
    switch (x->opcode()) {
      case HloOpcode::kBroadcast: {
        const auto& dims = x->dimensions();
        auto it = absl::c_find(dims, dim);
        HloInstruction* operand = x->mutable_operand(0);
        std::vector<int64_t> new_dims;
        if (it != dims.end()) {
          int64_t operand_dim = it - dims.begin();
          operand = Component(operand, operand_dim, k);
          for (int64_t i = 0; i < dims.size(); ++i) {
            if (i != operand_dim) {
              new_dims.push_back(dims[i] > dim ? dims[i] - 1 : dims[i]);
            }
          }
        } else {
          new_dims = DropDim(dims, dim);
        }
        return Add(HloInstruction::CreateBroadcast(squeezed, operand, new_dims));
      }
      case HloOpcode::kConcatenate: {
        const int64_t concat_dim = x->concatenate_dimension();
        if (concat_dim == dim) {
          int64_t offset = 0;
          for (HloInstruction* operand : x->operands()) {
            int64_t size = operand->shape().dimensions(dim);
            if (k < offset + size) {
              return Component(operand, dim, k - offset);
            }
            offset += size;
          }
          break;
        }
        std::vector<HloInstruction*> operands;
        for (HloInstruction* operand : x->operands()) {
          operands.push_back(Component(operand, dim, k));
        }
        return Add(HloInstruction::CreateConcatenate(
            squeezed, operands, concat_dim > dim ? concat_dim - 1 : concat_dim));
      }
      case HloOpcode::kSlice: {
        if (x->slice_strides(dim) <= 0) {
          break;
        }
        HloInstruction* operand = Component(
            x->mutable_operand(0), dim,
            x->slice_starts(dim) + k * x->slice_strides(dim));
        std::vector<int64_t> start, limit, strides;
        bool identity = true;
        for (int64_t i = 0; i < x->shape().dimensions().size(); ++i) {
          if (i == dim) {
            continue;
          }
          start.push_back(x->slice_starts(i));
          limit.push_back(x->slice_limits(i));
          strides.push_back(x->slice_strides(i));
          identity &= x->slice_starts(i) == 0 && x->slice_strides(i) == 1 &&
                      x->slice_limits(i) == x->operand(0)->shape().dimensions(i);
        }
        if (identity) {
          return operand;
        }
        return Add(HloInstruction::CreateSlice(squeezed, operand, start, limit,
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
        const int64_t index = k - pad.edge_padding_low();
        if (index < 0 || index >= operand->shape().dimensions(dim)) {
          return Add(HloInstruction::CreateBroadcast(
              squeezed, x->mutable_operand(1), {}));
        }
        PaddingConfig config;
        for (int64_t i = 0; i < x->shape().dimensions().size(); ++i) {
          if (i != dim) {
            *config.add_dimensions() = x->padding_config().dimensions(i);
          }
        }
        return Add(HloInstruction::CreatePad(
            squeezed, Component(operand, dim, index), x->mutable_operand(1),
            config));
      }
      case HloOpcode::kTranspose: {
        const int64_t operand_dim = x->dimensions(dim);
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < x->dimensions().size(); ++i) {
          if (i != dim) {
            int64_t p = x->dimensions(i);
            perm.push_back(p > operand_dim ? p - 1 : p);
          }
        }
        return Add(HloInstruction::CreateTranspose(
            squeezed, Component(x->mutable_operand(0), operand_dim, k), perm));
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
        HloInstruction* input = Component(x->mutable_operand(0), input_dim, k);
        return Add(HloInstruction::CreateReduce(
            squeezed, input, x->mutable_operand(1),
            DropDim(x->dimensions(), input_dim), x->to_apply()));
      }
      default:
        break;
    }
    return SliceAndSqueeze(x, dim, k);
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
  // Width-1 slices of a small dimension (e.g. the x, y or z component of a
  // [M, N, 3] array) are pushed up the same way, so that the 3-component array
  // is not needed by them.
  if (unroll_small_reductions_) {
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      ComponentHoister hoister(computation);
      for (HloInstruction* slice : computation->MakeInstructionPostOrder()) {
        if (slice->opcode() != HloOpcode::kSlice) {
          continue;
        }
        const Shape& in = slice->operand(0)->shape();
        int64_t dim = -1;
        bool simple = true;
        for (int64_t i = 0; i < in.dimensions().size(); ++i) {
          const bool full = slice->slice_starts(i) == 0 &&
                            slice->slice_strides(i) == 1 &&
                            slice->slice_limits(i) == in.dimensions(i);
          if (full) {
            continue;
          }
          if (dim != -1 || slice->shape().dimensions(i) != 1 ||
              in.dimensions(i) > max_kept_dim_size_) {
            simple = false;
          }
          dim = i;
        }
        const HloInstruction* operand = slice->operand(0);
        const bool hoistable =
            (operand->IsElementwise() &&
             operand->opcode() != HloOpcode::kConstant &&
             operand->operand_count() > 0) ||
            operand->opcode() == HloOpcode::kBroadcast ||
            operand->opcode() == HloOpcode::kConcatenate ||
            operand->opcode() == HloOpcode::kTranspose ||
            operand->opcode() == HloOpcode::kPad;
        // Hoisting through an op that is used elsewhere would duplicate it;
        // hoisting through an op the hoister cannot handle would only re-create
        // the slice.
        if (!simple || dim == -1 || !hoistable ||
            !absl::c_all_of(operand->users(), [](const HloInstruction* user) {
              return user->opcode() == HloOpcode::kSlice;
            })) {
          continue;
        }
        HloInstruction* component = hoister.Component(
            slice->mutable_operand(0), dim, slice->slice_starts(dim));
        HloInstruction* reshaped = computation->AddInstruction(
            HloInstruction::CreateReshape(slice->shape(), component));
        ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(slice, reshaped));
        changed = true;
      }
    }
  }
  // Reductions over one dimension of at most `max_kept_dim_size_` elements
  // become elementwise ops on the components, e.g. sum(d * d, axis=-1) over
  // xyz becomes d0 * d0 + d1 * d1 + d2 * d2.
  if (unroll_small_reductions_) {
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      for (HloInstruction* reduce : computation->MakeInstructionPostOrder()) {
        if (reduce->opcode() != HloOpcode::kReduce ||
            reduce->operand_count() != 2 || !reduce->shape().IsArray() ||
            reduce->dimensions().size() != 1) {
          continue;
        }
        const int64_t dim = reduce->dimensions(0);
        const int64_t size = reduce->operand(0)->shape().dimensions(dim);
        std::optional<HloOpcode> opcode =
            SimpleReducerOpcode(reduce->to_apply());
        if (size < 1 || size > max_kept_dim_size_ || !opcode.has_value()) {
          continue;
        }
        ComponentHoister hoister(computation);
        HloInstruction* acc = nullptr;
        for (int64_t i = 0; i < size; ++i) {
          HloInstruction* component =
              hoister.Component(reduce->mutable_operand(0), dim, i);
          acc = acc == nullptr
                    ? component
                    : computation->AddInstruction(HloInstruction::CreateBinary(
                          reduce->shape(), *opcode, acc, component));
        }
        // Include the init value, as the reduction does.
        HloInstruction* init = computation->AddInstruction(
            HloInstruction::CreateBroadcast(reduce->shape(),
                                            reduce->mutable_operand(1), {}));
        acc = computation->AddInstruction(HloInstruction::CreateBinary(
            reduce->shape(), *opcode, init, acc));
        ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(reduce, acc));
        changed = true;
      }
    }
  }
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
      if (reduced_elements < min_reduced_elements_ ||
          (variadic_is_faster_ &&
           !variadic_is_faster_(*reduce,
                                ShapeUtil::ElementsIn(reduce->shape()) / k))) {
        continue;
      }
      std::optional<HloOpcode> opcode =
          SimpleReducerOpcode(reduce->to_apply());
      if (!opcode.has_value()) {
        continue;
      }

      // Slices of the kept dimension, without that dimension.
      std::vector<HloInstruction*> slices;
      ComponentHoister hoister(computation);
      for (int64_t i = 0; i < k; ++i) {
        slices.push_back(hoister.Component(input, last, i));
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
