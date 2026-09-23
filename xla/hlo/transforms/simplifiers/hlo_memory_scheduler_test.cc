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

#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/memory_scheduler_metrics.pb.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class HloSchedulingTest : public HloHardwareIndependentTestBase {
 protected:
  AliasInfo alias_info_;
};

int64_t PeakMemoryUseOfEntryComputation(
    HloModule* module,
    const LogicalBuffer::SizeFunction* absl_nonnull size_function) {
  CHECK(module->has_entry_computation());
  CHECK(module->has_schedule());

  AliasInfo alias_info;
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module, &alias_info).value();

  const HloSchedule& schedule = module->schedule();

  HloComputation* computation = module->entry_computation();
  const HloInstructionSequence& sequence = schedule.sequence(computation);
  return HeapSimulator::Run(
             std::make_unique<NoFragmentationStatsHeap<HloValue>>(),
             *computation, sequence, *alias_analysis, &alias_info,
             size_function)
      .value()
      .heap_size;
}

TEST_F(HloSchedulingTest, LastUseScheduledFirst) {
  // Tests scheduling of the following HLO code:
  //
  //   %ab = abs(%param)
  //   %exp = exp(%param)
  //   %add = add(%ab, %exp)
  //   %negate = negate(%exp)
  //   %sub = subtract(%add, %negate)
  //
  // %add should be scheduled before %negate because %add is the last (and only)
  // use of %ab. Scheduling %add first then frees up %ab's buffer.
  const Shape vec = ShapeUtil::MakeShape(xla::F32, {42});
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec, "param"));
  auto ab = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kAbs, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kExp, param));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kAdd, ab, exp));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kNegate, exp));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kSubtract, add, negate));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloMemoryScheduler scheduler(&alias_info_, [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape());
  });
  ASSERT_FALSE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, scheduler.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());

  // Verify that all instructions are in the sequence.
  const std::vector<HloInstruction*>& sequence =
      module->schedule().sequence(module->entry_computation()).instructions();
  EXPECT_EQ(module->entry_computation()->instruction_count(), sequence.size());

  // The first instruction should be the parameter and the last the root "sub".
  EXPECT_EQ(param, sequence.front());
  EXPECT_EQ(sub, sequence.back());

  SequentialHloOrdering ordering(module->schedule());
  EXPECT_TRUE(ordering.ExecutesBefore(add, negate));

  // Clear the schedule using the descheduling pass.
  HloDescheduler descheduler;
  EXPECT_TRUE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(bool descheduler_changed,
                          descheduler.Run(module.get()));
  EXPECT_TRUE(descheduler_changed);
  EXPECT_FALSE(module->has_schedule());
}

TEST_F(HloSchedulingTest, ListSchedulerHandlesAliasing) {
  const char* module_str = R"(
HloModule test_aliasing_module

ENTRY root {
  param = s32[1000] parameter(0)
  p0 = s32[1000] copy(param)
  p1 = s32[1000] copy(param)
  t = (s32[1000], s32[1000]) tuple(p0, p1)
  a = s32[1000] get-tuple-element(t), index=0
  b = s32[1000] get-tuple-element(t), index=1
  c = s32[1000] add(a, b)
  d = s32[1000] add(c, b)
  e = s32[1000] add(c, c)
  f = s32[1000] add(e, e)
  ROOT result = (s32[1000], s32[1000], s32[1000]) tuple(d, e, f)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  BufferValue::SizeFunction size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  int64_t peak_memory;
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), ListMemoryScheduler(&alias_info_, &size_fn),
                     /*execution_threads=*/{}, &peak_memory));
  TF_ASSERT_OK(module->set_schedule(schedule));
  // Verify that all instructions are in the sequence.
  const std::vector<HloInstruction*>& sequence =
      schedule.sequence(module->entry_computation()).instructions();
  EXPECT_EQ(module->entry_computation()->instruction_count(), sequence.size());

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name;
  for (const HloInstruction* instruction : sequence) {
    instructions_by_name[instruction->name()] = instruction;
  }

  // The first instruction should be the parameter and the last the root.
  EXPECT_EQ(instructions_by_name.at("param"), sequence.front());
  EXPECT_EQ(instructions_by_name.at("result"), sequence.back());

  // Instructions "d" and "e" will both be schedulable at the same time, but
  // instruction "d" allows us to free the buffer of "p1", so the list scheduler
  // should prefer it.
  SequentialHloOrdering ordering(schedule);
  EXPECT_TRUE(ordering.ExecutesBefore(instructions_by_name.at("d"),
                                      instructions_by_name.at("e")));
  EXPECT_EQ(PeakMemoryUseOfEntryComputation(module.get(), &size_fn),
            peak_memory);
}

// Pins the list schedule of a tuple heavy while body. Its instructions read
// tuple elements through aliases, use the same buffer through two operands or
// twice through one, and use constants and parameters, whose buffers the
// heuristic ignores. Some ready instructions keep a stale priority until an
// instruction that shares an operand with them is scheduled, and dead
// instructions compete with the root. Counting a buffer twice, counting an
// ignored buffer as freed, refreshing priorities at other points, dropping
// the live out uses or reading only top level value sets each changes the
// order.
TEST_F(HloSchedulingTest, ListSchedulerTupleHeavyWhileBodyOrder) {
  const char* module_str = R"(
HloModule tuple_heavy_while

body {
  p = (f32[8], f32[16], f32[32], f32[64], s32[]) parameter(0)
  a = f32[8] get-tuple-element(p), index=0
  b = f32[16] get-tuple-element(p), index=1
  c = f32[32] get-tuple-element(p), index=2
  d = f32[64] get-tuple-element(p), index=3
  i = s32[] get-tuple-element(p), index=4
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  n_a = f32[8] negate(a)
  n_b = f32[16] negate(b)
  n_c = f32[32] negate(c)
  n_d = f32[64] negate(d)
  t = (f32[8], f32[16]) tuple(n_a, n_b)
  ta = f32[8] get-tuple-element(t), index=0
  tb = f32[16] get-tuple-element(t), index=1
  q = f32[4] slice(tb), slice={[0:4]}
  u = ((f32[8], f32[16]), f32[8]) tuple(t, ta)
  ug = (f32[8], f32[16]) get-tuple-element(u), index=0
  ua = f32[8] get-tuple-element(ug), index=0
  x = f32[8] add(ua, ta)
  y = f32[8] add(x, x)
  z = f32[16] add(tb, n_b)
  cat = f32[24] concatenate(y, z), dimensions={0}
  s_c = f32[24] slice(n_c), slice={[0:24]}
  ts = (f32[24]) tuple(s_c)
  gs = f32[24] get-tuple-element(ts), index=0
  k = f32[24] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23})
  r = f32[72] concatenate(k, gs, k), dimensions={0}
  s_d = f32[48] slice(n_d), slice={[8:56]}
  w = f32[96] concatenate(cat, s_c, s_d), dimensions={0}
  v = f32[96] multiply(w, w)
  sv = f32[24] slice(v), slice={[0:24]}
  vk = f32[24] add(sv, k)
  vk2 = f32[24] multiply(vk, vk)
  r_a = f32[8] slice(vk2), slice={[0:8]}
  dead = f32[4] slice(r_a), slice={[0:4]}
  r_b = f32[16] slice(vk), slice={[8:24]}
  big = f32[64] concatenate(r_b, r_b, r_b, r_b), dimensions={0}
  big2 = f32[32] concatenate(r_b, r_b), dimensions={0}
  r_d = f32[64] add(n_d, d)
  ROOT out = (f32[8], f32[16], f32[32], f32[64], s32[]) tuple(r_a, r_b, c, r_d, next_i)
}

cond {
  cond_p = (f32[8], f32[16], f32[32], f32[64], s32[]) parameter(0)
  cond_i = s32[] get-tuple-element(cond_p), index=4
  limit = s32[] constant(3)
  ROOT lt = pred[] compare(cond_i, limit), direction=LT
}

ENTRY main {
  pa = f32[8] parameter(0)
  pb = f32[16] parameter(1)
  pc = f32[32] parameter(2)
  pd = f32[64] parameter(3)
  zero = s32[] constant(0)
  e_b = f32[16] exponential(pb)
  e_d = f32[64] exponential(pd)
  init = (f32[8], f32[16], f32[32], f32[64], s32[]) tuple(pa, e_b, pc, e_d, zero)
  ROOT loop = (f32[8], f32[16], f32[32], f32[64], s32[]) while(init), condition=cond, body=body
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  BufferValue::SizeFunction size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     ListMemoryScheduler(&alias_info_, &size_fn)));

  auto names = [&](absl::string_view computation_name) {
    std::vector<std::string> result;
    for (const HloInstruction* instruction :
         schedule.sequence(FindComputation(module.get(), computation_name))
             .instructions()) {
      result.emplace_back(instruction->name());
    }
    return result;
  };
  EXPECT_THAT(names("body"),
              ::testing::ElementsAre("one", "p", "i", "next_i", "d", "c", "k",
                                     "b", "a", "n_a", "n_b", "t", "tb", "ta",
                                     "u", "ug", "ua", "x", "y", "q", "z", "cat",
                                     "n_c", "s_c", "ts", "gs", "n_d", "s_d",
                                     "r_d", "w", "v", "sv", "vk", "r_b", "vk2",
                                     "r_a", "dead", "out", "big2", "r", "big"));
  EXPECT_THAT(names("main"),
              ::testing::ElementsAre("zero", "pd", "pc", "pb", "pa", "e_b",
                                     "e_d", "init", "loop"));
}

TEST_F(HloSchedulingTest, DefaultSchedulerRunsThreeSchedulers) {
  const char* module_str = R"(
HloModule test_aliasing_module

ENTRY root {
  param = s32[1000] parameter(0)
  p0 = s32[1000] copy(param)
  p1 = s32[1000] copy(param)
  t = (s32[1000], s32[1000]) tuple(p0, p1)
  a = s32[1000] get-tuple-element(t), index=0
  b = s32[1000] get-tuple-element(t), index=1
  c = s32[1000] add(a, b)
  d = s32[1000] add(c, b)
  e = s32[1000] add(c, c)
  f = s32[1000] add(e, e)
  ROOT result = (s32[1000], s32[1000], s32[1000]) tuple(d, e, f)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));

  BufferValue::SizeFunction size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  module->metadata()->RecordPassStart();
  int64_t peak_memory;
  ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     DefaultMemoryScheduler(&alias_info_, &size_fn),
                     /*execution_threads=*/{}, &peak_memory));
  ASSERT_OK(module->set_schedule(schedule));
  MemorySchedulerMetrics metrics;
  auto metadata = module->metadata()->GetCurrentHloPassMetadata();
  ASSERT_OK(metadata.status());
  metadata.value()->custom_metadata().UnpackTo(&metrics);

  EXPECT_EQ(metrics.schedulers_size(), 3);

  auto list_metrics = metrics.schedulers(0);
  auto dfs_metrics = metrics.schedulers(1);
  auto post_order_metrics = metrics.schedulers(2);

  EXPECT_GT(list_metrics.peak_memory(), 0);
  EXPECT_GT(dfs_metrics.peak_memory(), 0);
  EXPECT_GT(post_order_metrics.peak_memory(), 0);

  EXPECT_TRUE(list_metrics.valid_schedule());
  EXPECT_TRUE(dfs_metrics.valid_schedule());
  EXPECT_TRUE(post_order_metrics.valid_schedule());

  EXPECT_GE(metrics.selected_scheduler_idx(), 0);
  EXPECT_LE(metrics.selected_scheduler_idx(), 2);
}

TEST_F(HloSchedulingTest, HostSendDoneSchedule) {
  const char* const module_str = R"(
HloModule module

ENTRY entry {
  %p = f32[1000, 1000] parameter(0)
  %token.0 = token[] after-all()
  %send = (f32[1000, 1000], token[]) send(%p, %token.0),
    channel_id=1, is_host_transfer=true
  %n1 = f32[1000, 1000] negate(%p)
  %n2 = f32[1000, 1000] negate(%n1)
  %n3 = f32[1000, 1000] negate(%n2)
  %send-done = token[] send-done(%send), channel_id=1, is_host_transfer=true
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), ListMemoryScheduler(&alias_info_, size_fn)));
  // Verify that all instructions are in the sequence.
  const std::vector<HloInstruction*>& sequence =
      schedule.sequence(module->entry_computation()).instructions();
  EXPECT_EQ(module->entry_computation()->instruction_count(), sequence.size());

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name;
  for (const HloInstruction* instruction : sequence) {
    instructions_by_name[instruction->name()] = instruction;
  }

  EXPECT_LT(absl::c_find(sequence, instructions_by_name.at("send-done")),
            absl::c_find(sequence, instructions_by_name.at("n1")));
}

TEST_F(HloSchedulingTest, TuplesAreAccountedCorrectly) {
  auto builder = HloComputation::Builder(TestName());
  const Shape r1f32 = ShapeUtil::MakeShape(xla::F32, {6});

  // Wrap lit in abs because constants are considered free by
  // IgnoreInstruction, and it skews the accounting.
  auto lit = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 1, 1, 1, 1, 1})));
  auto abs_const = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kAbs, lit));

  auto abs_abs1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kAbs, abs_const));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple(
      absl::Span<HloInstruction* const>({abs_abs1})));
  auto tuple_elm = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(r1f32, tuple, 0));

  auto abs_abs2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kAbs, abs_const));

  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd,
                                                      tuple_elm, abs_abs2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(
          module.get(),
          ListMemoryScheduler(&alias_info_, [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
          })));

  // Verify that all instructions are in the sequence.
  EXPECT_EQ(module->entry_computation()->instruction_count(),
            schedule.sequence(module->entry_computation()).size());
  SequentialHloOrdering ordering(schedule);
  // tuple allocates the tuple buffer and doesn't free anything.
  // abs_abs2 uses the same buffer for input/output, so its bytes-freed is 0.
  // abs_abs2 should be scheduled before tuple by List.
  EXPECT_TRUE(ordering.ExecutesBefore(abs_abs2, tuple));
}

TEST_F(HloSchedulingTest, MultiOutputFusionAccountedCorrectly) {
  const Shape r1f32 = ShapeUtil::MakeShape(xla::F32, {5});
  HloComputation::Builder builder(TestName());

  auto c1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 1, 1, 1, 1})));
  auto c2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5})));
  auto c3 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({0, 2, 4, 6, 8})));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd, c1, c2));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kMultiply, add, c3));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({add, mul}));

  auto tuple_elm = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(r1f32, tuple, 0));

  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kExp, c3));

  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd, tuple_elm, exp));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());

  auto fusion = computation->CreateFusionInstruction(
      {tuple, mul, add}, HloInstruction::FusionKind::kLoop);

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(
          module.get(),
          ListMemoryScheduler(&alias_info_, [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), 2);
          })));

  // Verify that all instructions are in the sequence.
  EXPECT_EQ(module->entry_computation()->instruction_count(),
            schedule.sequence(module->entry_computation()).size());
  SequentialHloOrdering ordering(schedule);
  // fusion allocates memory for the tuple elements and doesn't free anything,
  // so it's more expensive than exp.
  EXPECT_TRUE(ordering.ExecutesBefore(exp, fusion));
}

TEST_F(HloSchedulingTest, TrivialScheduler) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  param.b = (s32[], s32[]) parameter(0)
  gte.0 = s32[] get-tuple-element(param.b), index=0
  gte.1 = s32[] get-tuple-element(param.b), index=1
  add = s32[] add(gte.0, gte.1)
  ROOT tuple = (s32[], s32[]) tuple(gte.0, add)
}

cond {
  param.c = (s32[], s32[]) parameter(0)
  ROOT constant = pred[] constant(true)
}

ENTRY main {
  init = (s32[], s32[]) parameter(0)
  ROOT while = (s32[], s32[]) while(init), condition=cond, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(module->has_schedule());
  TF_ASSERT_OK(HloTrivialScheduler().Run(module.get()).status());
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());

  // Verify that a clone of the module also has a schedule.
  std::unique_ptr<HloModule> clone = module->Clone();
  ASSERT_TRUE(clone->has_schedule());
  TF_ASSERT_OK(clone->schedule().Verify());
}

TEST_F(HloSchedulingTest, BFSScheduler) {
  // When scheduling for maximum concurrency, we expect HLO operations to be
  // processed in wave-fronts: (1) all bcasts (2) followed by additions (3)
  // followed by reduction (4) final result accumulation. This would allow
  // XLA executor to overlap the execution of independent additions and
  // reductions at the cost of extra memory to keep temporaries alive.
  const char* const hlo_string = R"(
    HloModule m

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,512,256] parameter(0)
      c0 = f32[] constant(0)

      c1 = f32[] constant(1)
      bcast1 = f32[1,2,1,512,256] broadcast(c1), dimensions={}
      add1 = f32[1,2,1,512,256] add(p0, bcast1)

      c2 = f32[] constant(2)
      bcast2 = f32[1,2,1,512,256] broadcast(c2), dimensions={}
      add2 = f32[1,2,1,512,256] add(p0, bcast2)

      c3 = f32[] constant(3)
      bcast3 = f32[1,2,1,512,256] broadcast(c3), dimensions={}
      add3 = f32[1,2,1,512,256] add(p0, bcast3)

      c4 = f32[] constant(4)
      bcast4 = f32[1,2,1,512,256] broadcast(c4), dimensions={}
      add4 = f32[1,2,1,512,256] add(p0, bcast4)

      c5 = f32[] constant(5)
      bcast5 = f32[1,2,1,512,256] broadcast(c5), dimensions={}
      add5 = f32[1,2,1,512,256] add(p0, bcast5)

      r1 = f32[1,2] reduce(add1, c0), dimensions={2,3,4}, to_apply=add
      r2 = f32[1,2] reduce(add2, c0), dimensions={2,3,4}, to_apply=add
      r3 = f32[1,2] reduce(add3, c0), dimensions={2,3,4}, to_apply=add
      r4 = f32[1,2] reduce(add4, c0), dimensions={2,3,4}, to_apply=add
      r5 = f32[1,2] reduce(add5, c0), dimensions={2,3,4}, to_apply=add

      out0 = f32[1,2] add(r1, r2)
      out1 = f32[1,2] add(r3, r4)
      out2 = f32[1,2] add(out0, out1)
      ROOT out3 = f32[1,2] add(out2, r5)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     BFScheduler(&alias_info_, [](const BufferValue& buffer) {
                       return ShapeUtil::ByteSizeOf(buffer.shape());
                     })));

  const std::vector<HloInstruction*>& sequence =
      schedule.sequence(module->entry_computation()).instructions();

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name;
  for (const HloInstruction* instruction : sequence) {
    instructions_by_name[instruction->name()] = instruction;
  }

  auto index = [&](absl::string_view name) -> size_t {
    const HloInstruction* instruction = instructions_by_name.at(name);
    return std::distance(sequence.begin(), absl::c_find(sequence, instruction));
  };

  std::vector<size_t> indices = {
      index("bcast1"), index("bcast2"), index("bcast3"), index("bcast4"),
      index("bcast5"), index("add1"),   index("add2"),   index("add3"),
      index("add4"),   index("add5"),   index("r1"),     index("r2"),
      index("r3"),     index("r4"),     index("r5"),     index("out0"),
      index("out1"),   index("out2"),   index("out3")};

  EXPECT_TRUE(absl::c_is_sorted(indices));
}

}  // namespace
}  // namespace xla
