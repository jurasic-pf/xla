"""Biot-Savart VJP written with variadic reductions (one loop per pass, one
accumulator per component), to measure what XLA reaches when the per-pair work
is shared across the xyz outputs.

python gen_biot_variadic.py <n> : writes hlo/cpp/biot_vjp_variadic_<n>_f64.hlo
"""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

def sum_many(xs, axis):
    zeros = tuple(jnp.zeros((), x.dtype) for x in xs)
    return lax.reduce(tuple(xs), zeros, lambda a, b: tuple(p + q for p, q in zip(a, b)), (axis,))

def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

def vjp(src, ce, ev, g):
    s = [src[:, k][None, :] for k in range(3)]      # [1, N]
    c = [ce[:, k][None, :] for k in range(3)]
    e = [ev[:, k][:, None] for k in range(3)]       # [M, 1]
    gg = [g[:, k][:, None] for k in range(3)]
    d = [s[k] - e[k] for k in range(3)]              # [M, N], d_ji = src_j - ev_i
    inv_r = lax.rsqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    inv_r3 = inv_r * inv_r * inv_r
    inv_r5 = inv_r3 * inv_r * inv_r
    ce_x_g = cross(c, gg)
    d_x_ce = cross(d, c)
    g_dot = gg[0] * d_x_ce[0] + gg[1] * d_x_ce[1] + gg[2] * d_x_ce[2]
    t = [ce_x_g[k] * inv_r3 - 3.0 * g_dot * d[k] * inv_r5 for k in range(3)]
    g_x_d = cross(gg, d)
    u = [g_x_d[k] * inv_r3 for k in range(3)]
    grad_eval = sum_many([-x for x in t], axis=1)    # pass 1: over sources
    grad_rest = sum_many(t + u, axis=0)              # pass 2: over targets
    grad_src = jnp.stack(grad_rest[:3], axis=-1)
    grad_ce = jnp.stack(grad_rest[3:], axis=-1)
    return grad_src, grad_ce, jnp.stack(grad_eval, axis=-1)

n = int(sys.argv[1])
x = jnp.zeros((n, 3))
open(f"{os.environ.get('OUT', '.')}/biot_vjp_variadic_{n}_f64.hlo", "w").write(
    jax.jit(vjp).lower(x, x, x, x).compiler_ir(dialect="hlo").as_hlo_text())
print("wrote", n)
