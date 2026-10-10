/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_

#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/protobuf.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/custom_fusion_configs.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/backends/cpu/ynn_support.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/cpu_performance_model.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::cpu {

class YnnMatcher : public LibraryMatcher {
 public:
  explicit YnnMatcher(const TargetMachineFeatures* target_machine_features,
                      const tsl::protobuf::RepeatedField<int>* fusion_types)
      : LibraryMatcher(target_machine_features, fusion_types) {}
  ~YnnMatcher() override = default;

  // Returns the set of supported HLO instructions.
  absl::flat_hash_set<HloOpcode> SupportedOps() const override {
    static const absl::NoDestructor<absl::flat_hash_set<HloOpcode>>
        kSupportedOps{[]() {
          absl::flat_hash_set<HloOpcode> supported_ops{
              HloOpcode::kDot,          HloOpcode::kReduce,
              HloOpcode::kReduceWindow, HloOpcode::kConstant,
              HloOpcode::kConvolution,  HloOpcode::kReshape,
              HloOpcode::kBitcast,      HloOpcode::kBroadcast,
              HloOpcode::kTranspose,    HloOpcode::kPad,
              HloOpcode::kCopy,         HloOpcode::kIota};
          for (const auto& [op, _] : GetYnnUnaryOpMap()) {
            supported_ops.insert(op);
          }
          for (const auto& [op, _] : GetYnnBinaryOpMap()) {
            supported_ops.insert(op);
          }
          return supported_ops;
        }()};
    return *kSupportedOps;
  }

  // Returns true if the HLO instruction is supported by the library.
  absl::StatusOr<bool> IsOpSupported(const HloInstruction* instr) override {
    if (instr->IsConstant()) {
      return IsConstantSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kIota) {
      return IsIotaSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kReshape) {
      return IsReshapeOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kBitcast) {
      return IsBitcastOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kBroadcast) {
      return IsBroadcastOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kTranspose) {
      return IsTransposeOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kPad) {
      return IsPadOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kCopy) {
      return IsCopyOpSupportedByYnn(instr);
    }
    if (!IsInstructionPreferredByYnn(instr)) {
      // TODO: It might make sense sometimes that even though an instruction is
      // not preferred by YNNPACK, that we should still fuse it, if it lies
      // between two other fusions that are preferred. While it is currently
      // sometimes an advantage, it is also sometimes a regression, so for now,
      // we require every instruction to be preferred by YNNPACK.
      return false;
    }
    if (instr->opcode() == HloOpcode::kDot) {
      return IsDotSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kReduce ||
        instr->opcode() == HloOpcode::kReduceWindow) {
      return IsReduceLikeOpSupportedByYnn(instr);
    }
    if (instr->opcode() == HloOpcode::kConvolution) {
      return IsConvolutionOpSupportedByYnn(instr);
    }
    if (instr->IsElementwise()) {
      return IsElementwiseOpSupportedByYnn(instr);
    }
    return false;
  }

  // Returns true if we should start a new fusion containing just the given HLO
  // instruction. We control the instructions that can start a fusion with the
  // `--xla_cpu_experimental_ynn_fusion_type` flag.
  bool ShouldCreateFusion(const HloInstruction* instr) override {
    if (!IsInstructionPreferredByYnn(instr)) {
      return false;
    }
    if (fuse_dot_ && instr->opcode() == HloOpcode::kDot) {
      return true;
    }
    if (fuse_conv_ && instr->opcode() == HloOpcode::kConvolution) {
      return true;
    }
    if (fuse_reduce_ && (instr->opcode() == HloOpcode::kReduce ||
                         instr->opcode() == HloOpcode::kReduceWindow)) {
      return !LoopFusionIsFaster(instr);
    }
    return fuse_eltwise_ && instr->IsElementwise();
  }

  // Returns a prefix string for the fusion op's name.
  std::string fusion_prefix() const override { return "ynn_"; }

  // Returns a string for FusionBackendConfig's fusion kind.
  absl::string_view fusion_kind() const override { return kYnnFusionKind; }

  // The largest computed input of a reduction that a library fusion cannot
  // absorb, and whether it is in memory regardless of the fusion: a tuple
  // element, the root, or read by a user that a loop fusion cannot absorb.
  struct Materialized {
    const HloInstruction* instr = nullptr;
    bool anyway = false;
    // The other inputs that are not absorbed and not materialized anyway.
    std::vector<const HloInstruction*> others;
  };

  // A tiled library cannot stream through a transposition of a value it
  // computes: it writes the whole value and reads it back.
  static bool TransposesComputedValue(const HloInstruction& instr) {
    const bool transposes =
        instr.opcode() == HloOpcode::kTranspose ||
        (instr.opcode() == HloOpcode::kCopy && instr.shape().has_layout() &&
         instr.operand(0)->shape().has_layout() &&
         !LayoutUtil::Equal(instr.shape().layout(),
                            instr.operand(0)->shape().layout()));
    if (!transposes) {
      return false;
    }
    const HloInstruction* source = instr.operand(0);
    while (source->opcode() == HloOpcode::kBitcast) {
      source = source->operand(0);
    }
    return source->opcode() != HloOpcode::kParameter &&
           source->opcode() != HloOpcode::kConstant &&
           source->opcode() != HloOpcode::kGetTupleElement;
  }

  static bool IsMaterializedAnyway(
      const HloInstruction& instr,
      const absl::flat_hash_set<const HloInstruction*>& absorbed) {
    if (instr.opcode() == HloOpcode::kGetTupleElement ||
        instr.opcode() == HloOpcode::kParameter ||
        instr.opcode() == HloOpcode::kConstant ||
        (instr.parent() != nullptr &&
         instr.parent()->root_instruction() == &instr)) {
      return true;
    }
    return absl::c_any_of(instr.users(), [&](const HloInstruction* user) {
      if (absorbed.contains(user)) {
        return false;
      }
      switch (user->opcode()) {
        case HloOpcode::kBroadcast:
        case HloOpcode::kBitcast:
        case HloOpcode::kConcatenate:
        case HloOpcode::kDynamicSlice:
        case HloOpcode::kDynamicUpdateSlice:
        case HloOpcode::kPad:
        case HloOpcode::kReduce:
        case HloOpcode::kReduceWindow:
        case HloOpcode::kReshape:
        case HloOpcode::kReverse:
        case HloOpcode::kSlice:
        case HloOpcode::kTranspose:
          return false;
        case HloOpcode::kFusion:
          return !user->IsLoopFusion();
        default:
          return !user->IsElementwise();
      }
    });
  }

  // Returns the computed inputs of `reduce` that a fusion started at `reduce`
  // cannot absorb, e.g. because they have other users or are not supported.
  // The fusion reads them from memory, whereas a loop fusion can recompute
  // them.
  Materialized MaterializedInput(const HloInstruction* reduce) {
    // Follows the upward growth in LibraryRewriter::FuseNeighbors: an operand is
    // absorbed if it is supported and all of its users are absorbed.
    absl::flat_hash_set<const HloInstruction*> absorbed = {reduce};
    std::queue<const HloInstruction*> queue;
    queue.push(reduce);
    const HloInstruction* largest = nullptr;
    std::vector<const HloInstruction*> inputs;
    while (!queue.empty() &&
           static_cast<int64_t>(absorbed.size()) < MaxFusionSize()) {
      const HloInstruction* instr = queue.front();
      queue.pop();
      for (const HloInstruction* operand : instr->operands()) {
        if (absorbed.contains(operand) ||
            operand->opcode() == HloOpcode::kParameter ||
            operand->opcode() == HloOpcode::kConstant ||
            !operand->shape().IsArray()) {
          continue;
        }
        bool all_users_absorbed =
            absl::c_all_of(operand->users(), [&](const HloInstruction* user) {
              return absorbed.contains(user);
            });
        absl::StatusOr<bool> supported = IsOpSupported(operand);
        if (all_users_absorbed && supported.ok() && *supported &&
            !TransposesComputedValue(*operand)) {
          absorbed.insert(operand);
          queue.push(operand);
          continue;
        }
        if (!absl::c_linear_search(inputs, operand)) {
          inputs.push_back(operand);
        }
        if (largest == nullptr ||
            ShapeUtil::ByteSizeOfElements(operand->shape()) >
                ShapeUtil::ByteSizeOfElements(largest->shape())) {
          largest = operand;
        }
      }
    }
    if (largest == nullptr) {
      return {};
    }
    Materialized materialized{largest,
                              IsMaterializedAnyway(*largest, absorbed)};
    for (const HloInstruction* input : inputs) {
      if (input != largest && !IsMaterializedAnyway(*input, absorbed)) {
        materialized.others.push_back(input);
      }
    }
    return materialized;
  }

  // Returns true if a loop fusion with `outputs` outputs (default: those of
  // `reduce`) computes `reduce` faster than a library fusion that reads a
  // materialized input. Without such an input, the library fusion is kept.
  bool LoopFusionIsFaster(const HloInstruction* reduce,
                          int64_t outputs = -1) {
    return Decide(reduce, outputs).loop_fusion_is_faster;
  }

  // Returns true if `reduce` is left to a loop fusion that needs
  // TreeReductionRewriter to split it across threads.
  bool LoopFusionNeedsTreeReduction(const HloInstruction* reduce) {
    Decision decision = Decide(reduce, /*outputs=*/-1);
    return decision.loop_fusion_is_faster && decision.splits;
  }



 private:
  struct Decision {
    bool loop_fusion_is_faster = false;
    bool splits = false;
  };

  Decision Decide(const HloInstruction* reduce, int64_t outputs) {
    const Materialized materialized = MaterializedInput(reduce);
    if (materialized.instr == nullptr) {
      return {};
    }
    static const absl::NoDestructor<CpuPerformanceModel> model(
        CpuPerformanceModel::DefaultDeviceInfo());
    CpuPerformanceModel::ReduceWork work = CpuPerformanceModel::AnalyzeReduce(
        *reduce, materialized.instr, materialized.anyway, materialized.others);
    if (outputs > 0) {
      work.outputs = outputs;
    }
    CpuPerformanceModel::ReduceRunTimes times = model->EstimateReduce(work);
    VLOG(2) << reduce->name() << " materializing "
            << materialized.instr->name()
            << (materialized.anyway ? " (anyway)" : "")
            << ": " << work.ToString() << " loop fusion "
            << times.loop_fusion << " library " << times.library;
    return {times.loop_fusion < times.library,
            model->LoopFusionSplitsReduce(work)};
  }

  absl::flat_hash_set<DebugOptions::LibraryFusionType> fusion_types_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_YNN_MATCHER_H_
