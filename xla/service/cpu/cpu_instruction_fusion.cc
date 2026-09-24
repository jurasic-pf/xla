/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/cpu_instruction_fusion.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/cpu_hlo_cost_analysis.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/cpu_performance_model.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

namespace {

bool CanBeLoopFused(const HloInstruction& hlo) {
  // These are the only ones we fuse since we rely on effective elemental IR
  // generation.
  return hlo.IsElementwise() || hlo.opcode() == HloOpcode::kBitcast ||
         hlo.opcode() == HloOpcode::kBroadcast ||
         hlo.opcode() == HloOpcode::kConcatenate ||
         hlo.opcode() == HloOpcode::kDynamicSlice ||
         hlo.opcode() == HloOpcode::kDynamicUpdateSlice ||
         hlo.opcode() == HloOpcode::kGather ||
         hlo.opcode() == HloOpcode::kIota || hlo.opcode() == HloOpcode::kPad ||
         hlo.opcode() == HloOpcode::kReduce ||
         hlo.opcode() == HloOpcode::kReduceWindow ||
         hlo.opcode() == HloOpcode::kReshape ||
         hlo.opcode() == HloOpcode::kReverse ||
         hlo.opcode() == HloOpcode::kSlice ||
         hlo.opcode() == HloOpcode::kTranspose;
}

bool IsNonComplexNonBatchedMatrixVectorDot(const HloInstruction* hlo) {
  const Shape& hlo_shape = hlo->shape();
  return !ShapeUtil::ElementIsComplex(hlo_shape) &&
         hlo->opcode() == HloOpcode::kDot &&
         hlo_shape.dimensions().size() <= 1 &&
         hlo->dot_dimension_numbers().lhs_batch_dimensions_size() == 0;
}

bool HasExactlyOneUse(const HloInstruction& hlo_instr) {
  return hlo_instr.user_count() == 1 &&
         absl::c_count(hlo_instr.users().front()->operands(), &hlo_instr) == 1;
}

bool CanBeOutputFused(const HloInstruction* producer,
                      const HloInstruction* consumer) {
  return consumer->opcode() == HloOpcode::kAdd &&
         IsNonComplexNonBatchedMatrixVectorDot(producer) &&
         HasExactlyOneUse(*producer) == 1;
}

bool CanBeOutputFusedIntoSomeOperand(const HloInstruction* consumer) {
  return consumer->opcode() == HloOpcode::kAdd &&
         (CanBeOutputFused(consumer->operand(0), consumer) ||
          CanBeOutputFused(consumer->operand(1), consumer));
}

bool IsMaxReduction(const HloInstruction* reduce) {
  if (reduce->opcode() != HloOpcode::kReduce) {
    return false;
  }
  const HloInstruction* root = reduce->to_apply()->root_instruction();
  return root->opcode() == HloOpcode::kMaximum;
}

// Matches subtract(shift_value, broadcast(reduce)) or the reversed operand
// order subtract(broadcast(reduce), shift_value), used by exponential, i.e.
// the numerically sensitive stable-softmax shift pattern.
bool IsExpShiftCoupledSubtract(const HloInstruction& subtract,
                               const HloInstruction& shift_value,
                               const HloInstruction& reduce) {
  if (subtract.opcode() != HloOpcode::kSubtract) {
    return false;
  }
  const HloInstruction* op0 = subtract.operand(0);
  const HloInstruction* op1 = subtract.operand(1);
  const bool uses_shift_value = op0 == &shift_value || op1 == &shift_value ||
                                (op0->opcode() == HloOpcode::kBroadcast &&
                                 op0->operand(0) == &shift_value) ||
                                (op1->opcode() == HloOpcode::kBroadcast &&
                                 op1->operand(0) == &shift_value);
  const bool uses_reduce_broadcast =
      (op0->opcode() == HloOpcode::kBroadcast && op0->operand(0) == &reduce) ||
      (op1->opcode() == HloOpcode::kBroadcast && op1->operand(0) == &reduce);
  if (!uses_shift_value || !uses_reduce_broadcast) {
    return false;
  }
  return absl::c_any_of(subtract.users(), [](const HloInstruction* user) {
    return user->opcode() == HloOpcode::kExp;
  });
}

const HloInstruction* FindMaxReduceForShiftValue(
    const HloInstruction& shift_value) {
  for (const HloInstruction* user : shift_value.users()) {
    if (user->opcode() == HloOpcode::kReduce &&
        user->operand(0) == &shift_value && IsMaxReduction(user)) {
      return user;
    }
  }
  return nullptr;
}

// An elementwise op that feeds both row-max reduction and exp-shift
// subtraction. Matching any elementwise producer (not just multiply) covers
// numerically coupled shapes like add(dot, bias) as well as multiply(dot,
// scale), while staying within the reduce-max / exp-shift coupled structure.
bool IsCoupledReductionShiftExpProducer(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kFusion || !instr->IsElementwise()) {
    return false;
  }
  const HloInstruction* reduce = FindMaxReduceForShiftValue(*instr);
  if (reduce == nullptr) {
    return false;
  }
  return absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
    return IsExpShiftCoupledSubtract(*user, *instr, *reduce);
  });
}

// Should we block the fusion of the subcomputation of the passed instruction?
bool BlockSubcomputationFusion(const HloInstruction* instruction,
                               const HloModuleConfig& config) {
  HloOpcode opcode = instruction->opcode();
  if (opcode == HloOpcode::kScatter) {
    return true;
  }
  const bool use_experimental_fusion_emitters =
      options::UseExperimentalLoopFusion(config);

  // If the instruction itself can be fused then the subcomputation should be
  // blocked as the fusion emitter can't emit fusion ops inside another
  // fusion.
  if (use_experimental_fusion_emitters &&
      emitters::IsSupportedElementalOp(opcode)) {
    return true;
  }

  return false;
}

}  // namespace

bool CpuInstructionFusion::IsExpensive(const HloInstruction& instruction) {
  namespace m = match;

  switch (instruction.opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConstant:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kFloor:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kImag:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kMulhi:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kOr:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPad:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReplicaId:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSlice:
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
    case HloOpcode::kXor:
      return false;

    // Cheap instructions for reals, but expensive for complex.
    case HloOpcode::kAbs:
    case HloOpcode::kSign:
      return ShapeUtil::ElementIsComplex(instruction.shape());

    case HloOpcode::kConvert:
      // Converting from f32 to bf16 is expensive as we have to do multiple
      // checks for NaN, converting from bf16 to f32 is cheap as it is a simple
      // shift.
      return instruction.shape().element_type() == PrimitiveType::BF16 &&
             instruction.operand(0)->shape().element_type() ==
                 PrimitiveType::F32;

    // We say that integer div/mod by a constant is cheap because it gets
    // compiled down to multiplies and shifts, and we consider those to be
    // cheap.
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
      return !ShapeUtil::ElementIsIntegral(instruction.shape()) ||
             !Match(instruction.operand(0),
                    m::AnyOf<const HloInstruction>(
                        m::ConstantEffectiveScalar(),
                        m::Broadcast(m::ConstantEffectiveScalar())));

    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kTan:
      return ShapeUtil::ElementIsComplex(instruction.shape());

    case HloOpcode::kAcos:
    case HloOpcode::kAcosh:
    case HloOpcode::kSinh:
    case HloOpcode::kAsin:
    case HloOpcode::kAsinh:
    case HloOpcode::kAtan2:
    case HloOpcode::kAtanh:
    case HloOpcode::kCosh:
    case HloOpcode::kTanh:
      return true;

    case HloOpcode::kCbrt:
    case HloOpcode::kPower:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSqrt:
      return true;

    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
      return true;

      // Expensive instructions or unusual instructions for which fusion is
      // nonsensical.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kCall:
    case HloOpcode::kCholesky:
    case HloOpcode::kConditional:
    case HloOpcode::kConvolution:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectiveReduce:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kLogistic:
    case HloOpcode::kMap:
    case HloOpcode::kParameter:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kScaledDot:
    case HloOpcode::kScan:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kWhile:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return true;
  }

  return false;
}

InstructionFusion::HloInstructionSet
CpuInstructionFusion::ComputeGloballyUnfusible(
    absl::Span<HloInstruction* const> post_order,
    const HloReachabilityMap& reachability) {
  HloInstructionSet do_not_duplicate =
      InstructionFusion::ComputeGloballyUnfusible(post_order, reachability);
  for (HloInstruction* producer : post_order) {
    if (IsCoupledReductionShiftExpProducer(producer)) {
      // The base implementation treats effectively-unary elementwise ops as
      // free to duplicate. For producers feeding both a max-reduce and its
      // exp-shift subtract, duplication is unsound: FMA contraction can make
      // the recomputed copy differ from the copy the max was taken over by up
      // to one ulp, which exp() turns into inf/nan. Force such producers to be
      // materialized once.
      do_not_duplicate.insert(producer);
    }
  }
  return do_not_duplicate;
}

void CpuInstructionFusion::ComputeInstructionsToSkip(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto computations_list =
      module->MakeComputationPostOrder(execution_threads);
  instructions_to_skip_.clear();

  for (auto* computation : computations_list) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->IsCustomFusion()) {
        instructions_to_skip_.insert(instruction);
      } else if (instruction->opcode() == HloOpcode::kCustomCall) {
        HloCallableInstruction* callable =
            Cast<HloCallableInstruction>(instruction);
        if (callable->called_computations().empty()) {
          continue;
        }
        for (HloInstruction* instr :
             callable->called_computation()->instructions())
          instructions_to_skip_.insert(instr);
      } else if (BlockSubcomputationFusion(instruction, module->config())) {
        for (const auto* computation : instruction->called_computations()) {
          for (const auto* instr : computation->instructions()) {
            instructions_to_skip_.insert(instr);
          }
        }
      }
    }
  }
}

bool CpuInstructionFusion::ShouldSkip(const HloInstruction* inst) const {
  return instructions_to_skip_.contains(inst);
}

FusionDecision CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                                int64_t operand_index) {
  if (ShouldSkip(consumer)) {
    return FusionDecision::Forbid(
        "Don't fuse instructions from custom fusions/calls");
  }

  HloInstruction* producer = consumer->mutable_operand(operand_index);
  VLOG(2) << "Considering for fusion: operand " << operand_index << " of "
          << consumer->ToString();

  static constexpr int64_t kFusionThresholdBytes = 16 * 1024;

  // When we fuse a concatenate we don't take the fast path of simple memcpy /
  // for-loop; instead we currently emit a tree mapping the input to output idx
  // with a depth of log2(#args), this can have a large overhead for large
  // number of arguments.
  static constexpr int64_t kMaxConcatenateArguments = 8;

  if (HloPredicateIsOp<HloOpcode::kConstant>(producer) &&
      !ShapeUtil::IsEffectiveScalar(producer->shape())) {
    return FusionDecision::Forbid("Don't fuse non-scalar constants.");
  }

  if (CanBeOutputFused(producer, consumer)) {
    VLOG(2) << "Fusion OK: Can create output fusion.";
    return FusionDecision::Allow();
  }

  if (CanBeOutputFusedIntoSomeOperand(producer)) {
    return FusionDecision::Forbid(
        "Bailing because producer can be output-fused into some operand.");
  }

  if (!CanBeLoopFused(*producer)) {
    return FusionDecision::Forbid("Producer is not loop-fusible.");
  }

  // Concatenation on the minor dimension leads to inefficient code with a lot
  // of branches in the innermost loop. We prefer to materialize concatenated
  // buffers and run concat as a separate operation, as LLVM tends to do a
  // better job with pure data movement loops.
  auto is_minor_dim_concatenate = [](const HloInstruction* hlo) {
    // For vectors it's always beneficial to fuse concatenations.
    if (hlo->shape().dimensions().size() <= 1) {
      return false;
    }

    // Minor dimension concatenations with sufficient contiguous bytes benefit
    // from pure data movement (memcpy / SIMD loads and stores) when unfused.
    // Fusing them leads to branches in the innermost loop. However, small
    // concatenations (e.g. few rows or tiny total bytes) are dominated by
    // kernel launch and thunk dispatch overhead, so we keep them fused.
    int64_t concat_dim = hlo->concatenate_dimension();
    int64_t concat_dim_bytes =
        hlo->shape().dimensions(concat_dim) *
        ShapeUtil::ByteSizeOfPrimitiveType(hlo->shape().element_type());
    int64_t total_bytes = ShapeUtil::ByteSizeOfElements(hlo->shape());
    return concat_dim == LayoutUtil::Minor(hlo->shape().layout(), 0) &&
           concat_dim_bytes >= 64 && total_bytes >= 2048;
  };

  if ((producer->opcode() == HloOpcode::kConcatenate &&
       (producer->operand_count() > kMaxConcatenateArguments ||
        is_minor_dim_concatenate(producer))) ||
      (consumer->opcode() == HloOpcode::kConcatenate &&
       (consumer->operand_count() > kMaxConcatenateArguments ||
        is_minor_dim_concatenate(consumer)))) {
    return FusionDecision::Forbid("Concatenate fusion is inefficient.");
  }

  // Cost condition: not fuse (simple, expensive producers) and (consumers who
  // reuse operand elements).
  if (producer->opcode() != HloOpcode::kFusion &&
      ReusesOperandElements(consumer, operand_index) &&
      is_expensive(*producer)) {
    return FusionDecision::Forbid("Fusion is not profitable.");
  }

  RETURN_IF_NOT_FUSIBLE(InstructionFusion::ShouldFuse(consumer, operand_index));

  // Fusing too many reductions together can lead to a giant LLVM modules after
  // loop unrolling. We prefer to split such fusions into multiple kernels to
  // avoid excessive compilation times. X86TargetLowering::PerformDAGCombine
  // spends tens of minutes trying to combine load operations.
  //
  // TODO(b/419635451): Remove this once we have a better way to control the
  // size of the generated LLVM IR.
  static constexpr int64_t kMaxReductionsInFusion = 5;
  if (consumer->opcode() == HloOpcode::kFusion &&
      producer->opcode() == HloOpcode::kReduce) {
    int64_t num_fused_reductions = absl::c_count_if(
        consumer->fused_instructions(), [](const HloInstruction* instr) {
          return instr->opcode() == HloOpcode::kReduce;
        });
    if (num_fused_reductions > kMaxReductionsInFusion) {
      return FusionDecision::Forbid(
          "Too many reductions inside single fusion.");
    }
  }

  // Fuse constants in general but avoid creating 2-instruction fusions with
  // just a constant and another node.
  if (producer->opcode() == HloOpcode::kConstant &&
      consumer->opcode() != HloOpcode::kFusion) {
    return FusionDecision::Forbid(
        "Not fusing: insufficient non-constant nodes.");
  }

  // Output fusion is not currently supported on CPUs.
  if (producer->opcode() == HloOpcode::kFusion) {
    return FusionDecision::Forbid(
        "Not fusing: producer is itself a fusion node.");
  }

  // Don't fuse if fusing would cause too much code duplication because of
  // inefficiencies in the fusion emitter.
  // TODO(b/119692968): Remove this once the fusion emitter can handle
  // arbitrary fusion nodes.
  if (may_duplicate() && consumer->opcode() == HloOpcode::kFusion) {
    if (fusion_node_evaluations_.find(consumer) ==
        fusion_node_evaluations_.end()) {
      // We have no cached results for this fusion node yet. This can happen
      // when we run the InstructionFusion pass more than once. We can only
      // cache the results within one run.
      fusion_node_evaluations_.emplace(consumer,
                                       FusionNodeIndexingEvaluation(consumer));
    }
    const FusionNodeIndexingEvaluation& evaluation =
        fusion_node_evaluations_.at(consumer);
    // Below the limit, CodeDuplicationTooHigh only rejects emitting an op that
    // invalidates the elemental IR emitter's cache, e.g. a reduce, more than
    // once. Each copy recomputes the op, which the performance model accounts
    // for.
    if (evaluation.CodeDuplicationTooHigh(producer) &&
        (evaluation.EvaluateEmittedInstructions(producer) >
             FusionNodeIndexingEvaluation::kAllowedCodeDuplication ||
         !FusionIntoAllUsersIsFaster(*producer))) {
      return FusionDecision::Forbid("Code duplication too high");
    }
  }

  if (consumer->opcode() == HloOpcode::kDot) {
    // In the general case we call out to optimized "black box" GEMM routines
    // for Dot, which precludes fusion.  However, in very specific cases, we try
    // to fuse Dot operations by generating an elemental dot implementation.
    //
    // We need to be careful and conservative here since any benefit we get from
    // fusion can easily be overshadowed by the overhead of a naive GEMM
    // algorithm in the IR.
    const Shape& output_shape = consumer->shape();
    if (output_shape.dimensions().size() <= 1) {
      // We fuse in cases where we have a matrix*vector or vector*matrix dot and
      // fusion can get rid of the larger tensor.  We assume that a naive
      // traversal of a small enough (to fit in L1) column or row tensor is
      // "good enough" from the perspective of cache management; and calling out
      // to an optimized GEMM kernel is not a huge win.
      if (consumer->operand(0)->shape().dimensions().size() == 1 &&
          operand_index == 1 &&
          ShapeUtil::ByteSizeOfElements(consumer->operand(0)->shape()) <
              kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return FusionDecision::Allow();
      } else if (consumer->operand(1)->shape().dimensions().size() == 1 &&
                 operand_index == 0 &&
                 ShapeUtil::ByteSizeOfElements(consumer->operand(1)->shape()) <
                     kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return FusionDecision::Allow();
      }
    }
  }

  if (consumer->IsLoopFusion()) {
    VLOG(2) << "Fusing: consumer is a fusion node.";
    return FusionDecision::Allow();
  }

  if (CanBeLoopFused(*consumer)) {
    VLOG(2) << "Fusing: consumer is elementwise or fusible.";
    return FusionDecision::Allow();
  }

  return FusionDecision::Forbid("Not fusing: not found a fusible case");
}

HloInstruction::FusionKind CpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return CanBeOutputFused(producer, consumer)
             ? HloInstruction::FusionKind::kOutput
             : HloInstruction::FusionKind::kLoop;
}

absl::StatusOr<bool> CpuInstructionFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  fusion_node_evaluations_.clear();
  fusion_is_faster_.clear();
  ComputeInstructionsToSkip(module, execution_threads);

  HloCostAnalysis::Options options;
  options.count_multiple_input_accesses = true;
  cost_analysis_ = std::make_unique<CpuHloCostAnalysis>(options);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (absl::Status status = computation->Accept(cost_analysis_.get());
        !status.ok()) {
      VLOG(1) << "Cost analysis failed, falling back to IsExpensive: "
              << status;
      cost_analysis_.reset();
      break;
    }
  }
  set_is_expensive([this](const HloInstruction& instruction) {
    if (cost_analysis_ == nullptr) {
      return IsExpensive(instruction);
    }
    // Duplicating an expensive producer into several users gives each of its
    // fusible operands several users too, which the model does not account
    // for.
    if (IsExpensive(instruction) && instruction.user_count() > 1 &&
        absl::c_any_of(instruction.operands(), [](const HloInstruction* op) {
          return CanBeLoopFused(*op) && op->opcode() != HloOpcode::kBroadcast &&
                 op->opcode() != HloOpcode::kIota;
        })) {
      return true;
    }
    return !FusionIntoAllUsersIsFaster(instruction);
  });

  absl::StatusOr<bool> changed =
      InstructionFusion::RunImpl(module, execution_threads);
  cost_analysis_.reset();
  fusion_is_faster_.clear();
  return changed;
}

bool CpuInstructionFusion::FusionIntoAllUsersIsFaster(
    const HloInstruction& producer) {
  if (cost_analysis_ == nullptr || producer.user_count() == 0) {
    return false;
  }
  if (producer.opcode() == HloOpcode::kBroadcast ||
      producer.opcode() == HloOpcode::kIota ||
      ShapeUtil::IsEffectiveScalar(producer.shape())) {
    return true;
  }
  // A producer that fits into the cache is cheap to materialize, and the
  // estimate is dominated by the per-opcode flop counts, which do not account
  // for code generation costs.
  if (!producer.shape().IsArray() ||
      ShapeUtil::ByteSizeOfElements(producer.shape()) <
          performance_model_.device_info().l2_cache_size()) {
    return !IsExpensive(producer);
  }
  auto [it, inserted] =
      fusion_is_faster_.try_emplace(producer.unique_id(), false);
  if (!inserted) {
    return it->second;
  }
  it->second = EstimateFusionIntoAllUsersIsFaster(producer);
  return it->second;
}

bool CpuInstructionFusion::EstimateFusionIntoAllUsersIsFaster(
    const HloInstruction& producer) {
  // If any user is not fused, the producer is materialized anyway.
  for (const HloInstruction* user : producer.users()) {
    if (!user->IsLoopFusion() && !CanBeLoopFused(*user)) {
      return false;
    }
  }
  std::vector<const HloInstruction*> users(producer.users().begin(),
                                           producer.users().end());
  CpuPerformanceModel::RunTimes run_times = performance_model_.EstimateRunTimes(
      &producer, cost_analysis_.get(), users);
  // Operands used only by the producer are fused into it once if it is
  // materialized, and recomputed in each user if it is duplicated into fusions.
  if (producer.user_count() > 1 &&
      absl::c_all_of(producer.users(), [](const HloInstruction* user) {
        return user->opcode() == HloOpcode::kFusion;
      })) {
    run_times.time_fused += CpuPerformanceModel::ComputeTime(
        performance_model_.device_info(),
        (producer.user_count() - 1) * OperandChainFlops(producer));
  }
  return run_times.time_fused <= run_times.time_unfused;
}

int64_t CpuInstructionFusion::OperandChainFlops(
    const HloInstruction& producer) const {
  // An operand whose users are all in `chain` is fused into the producer if
  // the producer is materialized.
  absl::flat_hash_set<const HloInstruction*> chain = {&producer};
  std::vector<const HloInstruction*> worklist = {&producer};
  int64_t flops = 0;
  while (!worklist.empty()) {
    const HloInstruction* instr = worklist.back();
    worklist.pop_back();
    for (const HloInstruction* operand : instr->operands()) {
      if (chain.contains(operand) || !CanBeLoopFused(*operand) ||
          !absl::c_all_of(operand->users(), [&](const HloInstruction* user) {
            return chain.contains(user);
          })) {
        continue;
      }
      chain.insert(operand);
      worklist.push_back(operand);
      flops += std::max<int64_t>(0, cost_analysis_->flop_count(*operand));
    }
  }
  return flops;
}

HloInstruction* CpuInstructionFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  fusion_is_faster_.erase(producer->unique_id());
  HloInstruction* new_producer = FuseInstructionImpl(fusion_instruction,
                                                     producer);
  // The estimates for these instructions depend on the fused instructions or
  // on the users of the fused instructions.
  fusion_is_faster_.erase(fusion_instruction->unique_id());
  for (const HloInstruction* operand : fusion_instruction->operands()) {
    fusion_is_faster_.erase(operand->unique_id());
    for (const HloInstruction* operand_operand : operand->operands()) {
      fusion_is_faster_.erase(operand_operand->unique_id());
    }
  }
  if (cost_analysis_ != nullptr) {
    if (absl::Status status =
            cost_analysis_->RevisitInstruction(fusion_instruction);
        !status.ok()) {
      VLOG(1) << "Cost analysis failed, falling back to IsExpensive: "
              << status;
      cost_analysis_.reset();
    }
  }
  return new_producer;
}

HloInstruction* CpuInstructionFusion::FuseInstructionImpl(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  if (!may_duplicate()) {
    return InstructionFusion::FuseInstruction(fusion_instruction, producer);
  }

  auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
  if (evaluation == fusion_node_evaluations_.end()) {
    evaluation = fusion_node_evaluations_
                     .emplace(fusion_instruction,
                              FusionNodeIndexingEvaluation(fusion_instruction))
                     .first;
  }
  auto indexing_users = evaluation->second.RemoveFusionOperand(producer);
  HloInstruction* new_producer =
      InstructionFusion::FuseInstruction(fusion_instruction, producer);
  evaluation->second.UpdateEvaluationCache(new_producer, indexing_users);
  return new_producer;
}

}  // namespace cpu
}  // namespace xla
