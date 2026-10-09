"""Mean-field Kuramoto oscillators, fixed-step RK4 in lax.scan (plain JAX, .venv-libs).

Each RK4 stage reduces sin(theta_s) and cos(theta_s) over all N oscillators (order
parameter); the next stage depends on both sums. Two equivalent forms:
  polar:    omega + K r sin(psi - theta), r, psi from hypot/atan2 of the means
  cartmean: omega + K (S cos(theta) - C sin(theta)), S, C = jnp.mean(sin/cos)
Reproducer details and timings: repro_stage_reduce/.
"""
import os
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
OUT = os.environ.get("OUT", ".") + "/"


def kuramoto(theta, omega, form, steps=200, dt=0.01, K=1.5):
    def rhs(th):
        if form == "cartmean":
            s, c = jnp.mean(jnp.sin(th)), jnp.mean(jnp.cos(th))
            return omega + K * (s * jnp.cos(th) - c * jnp.sin(th))
        s = jnp.mean(jnp.sin(th), axis=-1, keepdims=True)
        c = jnp.mean(jnp.cos(th), axis=-1, keepdims=True)
        r, psi = jnp.hypot(s, c), jnp.arctan2(s, c)
        return omega + K * r * jnp.sin(psi - th)

    def step(th, _):
        k1 = rhs(th)
        k2 = rhs(th + 0.5 * dt * k1)
        k3 = rhs(th + 0.5 * dt * k2)
        k4 = rhs(th + dt * k3)
        return th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

    return jax.lax.scan(step, theta, None, length=steps)[0]


x = jnp.zeros(262144, jnp.float64)
for form in ("polar", "cartmean"):
    name = f"kuramoto_rk4_{form}_262144_f64"
    f = jax.jit(kuramoto, static_argnames="form")
    open(OUT + name + ".hlo", "w").write(f.lower(x, x, form=form).compiler_ir(dialect="hlo").as_hlo_text())
    print("wrote", name)
