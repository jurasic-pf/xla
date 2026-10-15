/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static void BM_ReduceAddF32(benchmark::State& state,
                            HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_add_f32_$d0

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[1,2] reduce(p0, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_ReduceAddBF16(benchmark::State& state,
                             HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_add_bf16_$d0

    add {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT add = bf16[] add(p0, p1)
    }

    ENTRY e {
      p0 = bf16[1,2,1,$d0,256] parameter(0)
      c0 = bf16[] constant(0)
      ROOT reduce = bf16[1,2] reduce(p0, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(BF16, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<BF16>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_ReduceAddF64(benchmark::State& state,
                            HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_add_f64_$d0

    add {
      p0 = f64[] parameter(0)
      p1 = f64[] parameter(1)
      ROOT add = f64[] add(p0, p1)
    }

    ENTRY e {
      p0 = f64[1,2,1,$d0,256] parameter(0)
      c0 = f64[] constant(0)
      ROOT reduce = f64[1,2] reduce(p0, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F64, {1, 2, 1, d0, 256});
  auto p0_status =
      LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  CHECK_OK(p0_status.status());

  std::vector<const Literal*> args = {&p0_status.value()};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_ReduceAddU32(benchmark::State& state,
                            HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_add_u32_$d0

    add {
      p0 = u32[] parameter(0)
      p1 = u32[] parameter(1)
      ROOT add = u32[] add(p0, p1)
    }

    ENTRY e {
      p0 = u32[1,2,1,$d0,256] parameter(0)
      c0 = u32[] constant(0)
      ROOT reduce = u32[1,2] reduce(p0, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  Shape shape = ShapeUtil::MakeShape(U32, {1, 2, 1, d0, 256});
  auto p0_or = LiteralUtil::CreateRandomLiteral<U32>(shape, &engine, 100, 1);
  CHECK_OK(p0_or.status());
  Literal p0 = std::move(p0_or).value();

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_ReduceAddU64(benchmark::State& state,
                            HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule reduce_add_u64_$d0

    add {
      p0 = u64[] parameter(0)
      p1 = u64[] parameter(1)
      ROOT add = u64[] add(p0, p1)
    }

    ENTRY e {
      p0 = u64[1,2,1,$d0,256] parameter(0)
      c0 = u64[] constant(0)
      ROOT reduce = u64[1,2] reduce(p0, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  Shape shape = ShapeUtil::MakeShape(U64, {1, 2, 1, d0, 256});
  auto p0_or = LiteralUtil::CreateRandomLiteral<U64>(shape, &engine, 100, 1);
  CHECK_OK(p0_or.status());
  Literal p0 = std::move(p0_or).value();

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_SumOfSquaresF32(benchmark::State& state,
                               HloBenchmarkOptions options) {
  int64_t d0 = state.range(0);

  absl::string_view hlo = R"(
    HloModule sum_of_squares_f32_$d0

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      c0 = f32[] constant(0)
      mul = f32[1,2,1,$d0,256] multiply(p0, p0)
      ROOT reduce = f32[1,2] reduce(mul, c0), dimensions={2,3,4}, to_apply=add
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}, options));
}

static void BM_ReduceWindowAddF32OuterAndInnerDim(benchmark::State& state,
                                                  HloBenchmarkOptions options) {
  int outer_dim = state.range(0);
  int inner_dim = state.range(1);

  constexpr absl::string_view hlo = R"(
  HloModule reduce_window_add_f32_outer_dim_$outer_dim_inner_dim_$inner_dim

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[1024,1024] parameter(0)
    c0 = f32[] constant(0)
    ROOT reduce = f32[$result_outer_dim,$result_inner_dim] 
      reduce-window(p0, c0), window={size=$outer_dimx$inner_dim stride=$outer_dimx$inner_dim}, to_apply=add
  }
)";

  CHECK_OK(
      RunHloBenchmark(state, hlo, {},
                      {
                          {"$outer_dim", absl::StrCat(outer_dim)},
                          {"$result_outer_dim", absl::StrCat(1024 / outer_dim)},
                          {"$inner_dim", absl::StrCat(inner_dim)},
                          {"$result_inner_dim", absl::StrCat(1024 / inner_dim)},
                      },
                      options));
}

static void BM_ReduceWindowAddBF16OuterAndInnerDim(
    benchmark::State& state, HloBenchmarkOptions options) {
  int outer_dim = state.range(0);
  int inner_dim = state.range(1);

  constexpr absl::string_view hlo = R"(
  HloModule reduce_window_add_bf16_outer_dim_$outer_dim_inner_dim_$inner_dim

  add {
    p0 = bf16[] parameter(0)
    p1 = bf16[] parameter(1)
    ROOT add = bf16[] add(p0, p1)
  }

  ENTRY e {
    p0 = bf16[1024,1024] parameter(0)
    c0 = bf16[] constant(0)
    ROOT reduce = bf16[$result_outer_dim,$result_inner_dim]
      reduce-window(p0, c0), window={size=$outer_dimx$inner_dim stride=$outer_dimx$inner_dim}, to_apply=add
  }
)";

  CHECK_OK(
      RunHloBenchmark(state, hlo, {},
                      {
                          {"$outer_dim", absl::StrCat(outer_dim)},
                          {"$result_outer_dim", absl::StrCat(1024 / outer_dim)},
                          {"$inner_dim", absl::StrCat(inner_dim)},
                          {"$result_inner_dim", absl::StrCat(1024 / inner_dim)},
                      },
                      options));
}

static void BM_ReduceAddF32OverDimension(benchmark::State& state,
                                         HloBenchmarkOptions options) {
  int64_t reduce_dim = state.range(0);

  constexpr absl::string_view hlo = R"(
  HloModule reduce_add_f32_reduce_dim_$reduce_dim

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[1024,1024] parameter(0)
    c0 = f32[] constant(0)
    ROOT reduce = f32[1024] reduce(p0, c0), dimensions={$reduce_dim}, to_apply=add
  }
)";

  CHECK_OK(RunHloBenchmark(
      state, hlo, {}, {{"$reduce_dim", absl::StrCat(reduce_dim)}}, options));
}

static void BM_ReduceAddBF16OverDimension(benchmark::State& state,
                                          HloBenchmarkOptions options) {
  int64_t reduce_dim = state.range(0);

  constexpr absl::string_view hlo = R"(
  HloModule reduce_add_bf16_reduce_dim_$reduce_dim

  add {
    p0 = bf16[] parameter(0)
    p1 = bf16[] parameter(1)
    ROOT add = bf16[] add(p0, p1)
  }

  ENTRY e {
    p0 = bf16[1024,1024] parameter(0)
    c0 = bf16[] constant(0)
    ROOT reduce = bf16[1024] reduce(p0, c0), dimensions={$reduce_dim}, to_apply=add
  }
)";

  CHECK_OK(RunHloBenchmark(
      state, hlo, {}, {{"$reduce_dim", absl::StrCat(reduce_dim)}}, options));
}

static void BM_ReduceAddF64OverDimension(benchmark::State& state,
                                         HloBenchmarkOptions options) {
  int64_t reduce_dim = state.range(0);

  constexpr absl::string_view hlo = R"(
  HloModule reduce_add_f64_reduce_dim_$reduce_dim

  add {
    p0 = f64[] parameter(0)
    p1 = f64[] parameter(1)
    ROOT add = f64[] add(p0, p1)
  }

  ENTRY e {
    p0 = f64[1024,1024] parameter(0)
    c0 = f64[] constant(0)
    ROOT reduce = f64[1024] reduce(p0, c0), dimensions={$reduce_dim}, to_apply=add
  }
)";

  CHECK_OK(RunHloBenchmark(
      state, hlo, {}, {{"$reduce_dim", absl::StrCat(reduce_dim)}}, options));
}

static void BM_ReduceWindowAddF32SkippingData(benchmark::State& state,
                                              HloBenchmarkOptions options) {
  constexpr absl::string_view hlo = R"(
  HloModule reduce_window_add_f32_skipping_data

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[128,128] parameter(0)
    c0 = f32[] constant(0)
  ROOT reduce = f32[128,2] reduce-window(p0, c0), window={size=1x8 stride=1x64}, to_apply=add
  }
  )";

  CHECK_OK(RunHloBenchmark(state, hlo, {}, {{}}, options));
}

static void BM_ReduceWindowAddF32OverlappingWindows(
    benchmark::State& state, HloBenchmarkOptions options) {
  constexpr absl::string_view hlo = R"(
  HloModule reduce_window_add_f32_overlapping_windows

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT add = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[128,128] parameter(0)
    c0 = f32[] constant(0)
    ROOT reduce = f32[128,13] reduce-window(p0, c0), window={size=1x32 stride=1x8}, to_apply=add
  }
  )";

  CHECK_OK(RunHloBenchmark(state, hlo, {}, {{}}, options));
}

// Mean-field Kuramoto oscillators integrated with RK4 inside a while loop:
//   dtheta/dt = omega + K * (S * cos(theta) - C * sin(theta)),
//   S = mean(sin(theta)), C = mean(cos(theta)).
// Each RK4 stage reduces sin(theta_s) and cos(theta_s) over all oscillators,
// and the next stage depends on both sums. With `mean` set, the sums are
// scaled before they are broadcast (as jnp.mean does), otherwise after.
static void BM_KuramotoRk4F32(benchmark::State& state,
                              HloBenchmarkOptions options) {
  int64_t n = state.range(0);
  bool mean = state.range(1);

  absl::string_view hlo = R"(
    HloModule kuramoto_rk4_f32_$n

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    rhs {
      theta = f32[$n] parameter(0)
      omega = f32[$n] parameter(1)
      c0 = f32[] constant(0)
      sin = f32[$n] sine(theta)
      cos = f32[$n] cosine(theta)
      sum_sin = f32[] reduce(sin, c0), dimensions={0}, to_apply=add
      sum_cos = f32[] reduce(cos, c0), dimensions={0}, to_apply=add
      $coupling
      ROOT dtheta = f32[$n] add(omega, coupling)
    }

    body {
      state = (s32[], f32[$n], f32[$n]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      theta = f32[$n] get-tuple-element(state), index=1
      omega = f32[$n] get-tuple-element(state), index=2
      half_dt = f32[] constant(0.005)
      dt = f32[] constant(0.01)
      dt_6 = f32[] constant(0.0016666667)
      two = f32[] constant(2)
      half_dt_b = f32[$n] broadcast(half_dt), dimensions={}
      dt_b = f32[$n] broadcast(dt), dimensions={}
      dt_6_b = f32[$n] broadcast(dt_6), dimensions={}
      two_b = f32[$n] broadcast(two), dimensions={}
      k1 = f32[$n] call(theta, omega), to_apply=rhs
      d1 = f32[$n] multiply(half_dt_b, k1)
      theta1 = f32[$n] add(theta, d1)
      k2 = f32[$n] call(theta1, omega), to_apply=rhs
      d2 = f32[$n] multiply(half_dt_b, k2)
      theta2 = f32[$n] add(theta, d2)
      k3 = f32[$n] call(theta2, omega), to_apply=rhs
      d3 = f32[$n] multiply(dt_b, k3)
      theta3 = f32[$n] add(theta, d3)
      k4 = f32[$n] call(theta3, omega), to_apply=rhs
      k2_2 = f32[$n] multiply(two_b, k2)
      k3_2 = f32[$n] multiply(two_b, k3)
      s12 = f32[$n] add(k1, k2_2)
      s123 = f32[$n] add(s12, k3_2)
      s1234 = f32[$n] add(s123, k4)
      d = f32[$n] multiply(dt_6_b, s1234)
      theta_next = f32[$n] add(theta, d)
      one = s32[] constant(1)
      i_next = s32[] add(i, one)
      ROOT result = (s32[], f32[$n], f32[$n]) tuple(i_next, theta_next, omega)
    }

    cond {
      state = (s32[], f32[$n], f32[$n]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      steps = s32[] constant(10)
      ROOT lt = pred[] compare(i, steps), direction=LT
    }

    ENTRY e {
      theta = f32[$n] parameter(0)
      omega = f32[$n] parameter(1)
      zero = s32[] constant(0)
      init = (s32[], f32[$n], f32[$n]) tuple(zero, theta, omega)
      loop = (s32[], f32[$n], f32[$n]) while(init), condition=cond, body=body
      ROOT out = f32[$n] get-tuple-element(loop), index=1
    }
  )";

  // K * (S * cos - C * sin), scaling the scalar sums.
  absl::string_view scale_sums = R"(
      n = f32[] constant($n)
      mean_sin = f32[] divide(sum_sin, n)
      mean_cos = f32[] divide(sum_cos, n)
      k = f32[] constant(1.5)
      k_sin = f32[] multiply(k, mean_sin)
      k_cos = f32[] multiply(k, mean_cos)
      k_sin_b = f32[$n] broadcast(k_sin), dimensions={}
      k_cos_b = f32[$n] broadcast(k_cos), dimensions={}
      a = f32[$n] multiply(k_sin_b, cos)
      b = f32[$n] multiply(k_cos_b, sin)
      coupling = f32[$n] subtract(a, b))";

  // K / N * (S * cos - C * sin), scaling after the broadcast.
  absl::string_view scale_products = R"(
      sum_sin_b = f32[$n] broadcast(sum_sin), dimensions={}
      sum_cos_b = f32[$n] broadcast(sum_cos), dimensions={}
      a = f32[$n] multiply(sum_sin_b, cos)
      b = f32[$n] multiply(sum_cos_b, sin)
      diff = f32[$n] subtract(a, b)
      k_over_n = f32[] constant($k_over_n)
      k_over_n_b = f32[$n] broadcast(k_over_n), dimensions={}
      coupling = f32[$n] multiply(k_over_n_b, diff))";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {n});
  auto theta =
      *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 0.0f, 1.0f);
  auto omega =
      *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 0.0f, 1.0f);

  std::string hlo_text = absl::StrReplaceAll(
      hlo, {{"$coupling", mean ? scale_sums : scale_products}});

  std::vector<const Literal*> args = {&theta, &omega};
  CHECK_OK(RunHloBenchmark(
      state, hlo_text, args,
      {{"$n", absl::StrCat(n)}, {"$k_over_n", absl::StrCat(1.5 / n)}},
      options));
}

#define BENCHMARK_SIZES(NAME)   \
  XLA_CPU_BENCHMARK(NAME)       \
      ->MeasureProcessCPUTime() \
      ->Arg(128)                \
      ->Arg(256)                \
      ->Arg(512)                \
      ->Arg(1024)               \
      ->Arg(8192)               \
      ->Arg(16384)              \
      ->MeasureProcessCPUTime();

BENCHMARK_SIZES(BM_ReduceAddF32);
BENCHMARK_SIZES(BM_ReduceAddBF16);
BENCHMARK_SIZES(BM_ReduceAddF64);
BENCHMARK_SIZES(BM_ReduceAddU32);
BENCHMARK_SIZES(BM_ReduceAddU64);
BENCHMARK_SIZES(BM_SumOfSquaresF32);

XLA_CPU_BENCHMARK(BM_ReduceAddF32OverDimension)
    ->ArgName("reduce_dim")
    ->Arg(0)
    ->Arg(1)
    ->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_ReduceAddBF16OverDimension)
    ->ArgName("reduce_dim")
    ->Arg(0)
    ->Arg(1)
    ->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_ReduceAddF64OverDimension)
    ->ArgName("reduce_dim")
    ->Arg(0)
    ->Arg(1)
    ->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_ReduceWindowAddF32OuterAndInnerDim)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "inner_dim"})
    ->Args({1, 32})
    ->Args({32, 1})
    ->Args({32, 2})
    ->Args({32, 4})
    ->Args({32, 8})
    ->Args({32, 16})
    ->Args({32, 32})
    ->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_ReduceWindowAddBF16OuterAndInnerDim)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "inner_dim"})
    ->Args({1, 32})
    ->Args({32, 1})
    ->Args({32, 2})
    ->Args({32, 4})
    ->Args({32, 8})
    ->Args({32, 16})
    ->Args({32, 32})
    ->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_ReduceWindowAddF32SkippingData)->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_ReduceWindowAddF32OverlappingWindows)
    ->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_KuramotoRk4F32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"n", "mean"})
    ->ArgsProduct({{1 << 16, 1 << 18, 1 << 20}, {0, 1}});

}  // namespace xla::cpu
