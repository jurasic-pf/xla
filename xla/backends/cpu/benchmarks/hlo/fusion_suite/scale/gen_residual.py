"""Residual-stream transformer skeleton for compile-time scaling: L layers of
x = bf16(f32(x) + f32(dot(bf16(rmsnorm(f32(x))), W_k))) on a small [T,D] bf16 stream, as in Gemma.
usage: gen_residual.py L [T] [D] > residual_L.hlo"""
import sys
L = int(sys.argv[1]); T = int(sys.argv[2]) if len(sys.argv) > 2 else 11; D = int(sys.argv[3]) if len(sys.argv) > 3 else 1152
o = [f"HloModule residual_{L}", "",
     "add_f32 {", "  a = f32[] parameter(0)", "  b = f32[] parameter(1)", "  ROOT s = f32[] add(a, b)", "}", "",
     "ENTRY main {", f"  x0 = bf16[{T},{D}]{{1,0}} parameter(0)", "  zero = f32[] constant(0)",
     f"  invd = f32[] constant({1.0 / D})", "  eps = f32[] constant(1e-6)"]
x = "x0"
for k in range(L):
    p = f"l{k}_"
    o += [f"  {p}w = bf16[{D},{D}]{{1,0}} parameter({k + 1})",
          f"  {p}xf = f32[{T},{D}]{{1,0}} convert({x})",
          f"  {p}sq = f32[{T},{D}]{{1,0}} multiply({p}xf, {p}xf)",
          f"  {p}ss = f32[{T}]{{0}} reduce({p}sq, zero), dimensions={{1}}, to_apply=add_f32",
          f"  {p}bi = f32[{T}]{{0}} broadcast(invd), dimensions={{}}",
          f"  {p}ms = f32[{T}]{{0}} multiply({p}ss, {p}bi)",
          f"  {p}be = f32[{T}]{{0}} broadcast(eps), dimensions={{}}",
          f"  {p}me = f32[{T}]{{0}} add({p}ms, {p}be)",
          f"  {p}r = f32[{T}]{{0}} rsqrt({p}me)",
          f"  {p}br = f32[{T},{D}]{{1,0}} broadcast({p}r), dimensions={{0}}",
          f"  {p}n = f32[{T},{D}]{{1,0}} multiply({p}xf, {p}br)",
          f"  {p}nb = bf16[{T},{D}]{{1,0}} convert({p}n)",
          f"  {p}y = bf16[{T},{D}]{{1,0}} dot({p}nb, {p}w), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}",
          f"  {p}yf = f32[{T},{D}]{{1,0}} convert({p}y)",
          f"  {p}xs = f32[{T},{D}]{{1,0}} add({p}xf, {p}yf)",
          f"  {p}x = bf16[{T},{D}]{{1,0}} convert({p}xs)"]
    x = f"{p}x"
o += [f"  ROOT out = bf16[{T},{D}]{{1,0}} copy({x})", "}"]
print("\n".join(o))
