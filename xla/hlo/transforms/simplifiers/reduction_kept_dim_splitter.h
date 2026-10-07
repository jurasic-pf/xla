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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCTION_KEPT_DIM_SPLITTER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCTION_KEPT_DIM_SPLITTER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Splits a reduction that keeps a small last dimension into one variadic
// reduction over the slices of that dimension, e.g.
//
//   r = f32[M,3] reduce(f32[M,N,3] x, c), dimensions={1}
//
// becomes
//
//   s_k = f32[M,N] reshape(slice(x, [.., .., k:k+1])), k = 0, 1, 2
//   t = (f32[M], f32[M], f32[M]) reduce(s_0, s_1, s_2, c, c, c), dimensions={1}
//   r = f32[M,3] concatenate(reshape(get-tuple-element(t, k)))
//
// A loop emitter iterates the kept dimension outside the reduced ones, so the
// original reduction computes the producer of `x` once per slice and selects
// the slice inside the loop. The variadic reduction computes all slices in one
// pass with one accumulator each, and the slice index is a constant in each.
class ReductionKeptDimSplitter : public HloModulePass {
 public:
  explicit ReductionKeptDimSplitter(int64_t max_kept_dim_size = 8,
                                    int64_t min_reduced_elements = 1024,
                                    bool unroll_small_reductions = false)
      : max_kept_dim_size_(max_kept_dim_size),
        min_reduced_elements_(min_reduced_elements),
        unroll_small_reductions_(unroll_small_reductions) {}
  absl::string_view name() const override {
    return "reduction-kept-dim-splitter";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t max_kept_dim_size_;
  int64_t min_reduced_elements_;
  bool unroll_small_reductions_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCTION_KEPT_DIM_SPLITTER_H_
