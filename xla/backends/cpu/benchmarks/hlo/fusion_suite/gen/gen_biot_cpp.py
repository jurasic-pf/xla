"""Biot-Savart VJP matching the hand-written C++ kernel (all three inputs, f64).

python gen_biot_cpp.py <n> : writes hlo/cpp/biot_vjp_<n>_f64.hlo and biot_fwd_<n>_f64.hlo
"""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def field(src, ce, ev):
    d = src[None, :, :] - ev[:, None, :]          # [M, N, 3], d_ji = src_j - ev_i
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    return jnp.sum(jnp.cross(d, ce[None, :, :]) / r[..., None] ** 3, axis=1)

def vjp(src, ce, ev, g):
    _, pullback = jax.vjp(field, src, ce, ev)
    return pullback(g)

n = int(sys.argv[1])
x = jnp.zeros((n, 3))
out = os.environ.get("OUT", ".") + "/"
open(out + f"biot_vjp_{n}_f64.hlo", "w").write(jax.jit(vjp).lower(x, x, x, x).compiler_ir(dialect="hlo").as_hlo_text())
open(out + f"biot_fwd_{n}_f64.hlo", "w").write(jax.jit(field).lower(x, x, x).compiler_ir(dialect="hlo").as_hlo_text())
print("wrote", n)
