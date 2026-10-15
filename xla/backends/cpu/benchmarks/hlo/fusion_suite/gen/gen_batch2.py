"""Second batch: diffrax stiff ODE, jax-md Lennard-Jones, large-vocab
cross-entropy, Llama-style RMSNorm + SwiGLU block (fwd and grad)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_lapack  # noqa: F401
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
OUT = os.environ.get("OUT", ".") + "/"

def dump(name, f, *args):
    txt = jax.jit(f).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    cc = sorted(set(__import__("re").findall(r"custom_call_target=\"([^\"]*)\"", txt)))
    if cc: print("SKIP", name, cc); return
    open(OUT + name + ".hlo", "w").write(txt); print("wrote", name)

# diffrax: Robertson chemical kinetics (stiff), batched over 256 initial
# conditions, Kvaerno5 with PID step control, 64 saved points.
import diffrax
def robertson(t, y, args):
    k1, k2, k3 = 0.04, 3e7, 1e4
    return jnp.stack([-k1 * y[0] + k3 * y[1] * y[2],
                      k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
                      k2 * y[1] ** 2])
def solve(y0):
    sol = diffrax.diffeqsolve(diffrax.ODETerm(robertson), diffrax.Kvaerno5(), 0., 100., 1e-4, y0,
                              stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
                              saveat=diffrax.SaveAt(ts=jnp.linspace(0., 100., 64)), max_steps=4096)
    return sol.ys
y0 = jnp.tile(jnp.array([1., 0., 0.]), (256, 1)) * jnp.linspace(0.5, 1.5, 256)[:, None]
dump("diffrax_robertson_kvaerno5", jax.vmap(solve), y0)

# jax-md: Lennard-Jones energy and forces, 4096 particles, periodic box,
# all pairs (no neighbor list) and with a cell neighbor list.
from jax_md import space, energy, quantity
N, box = 4096, 18.0
R = jax.random.uniform(jax.random.PRNGKey(0), (N, 3), maxval=box)
disp, shift = space.periodic(box)
lj = energy.lennard_jones_pair(disp, sigma=1.0, epsilon=1.0, r_onset=2.0, r_cutoff=2.5)
dump("jaxmd_lj_allpairs_energy", lj, R)
dump("jaxmd_lj_allpairs_force", quantity.force(lj), R)

# Large-vocab softmax cross-entropy + grad (f32), batch 2048, vocab 32000.
jax.config.update("jax_enable_x64", False)
logits = jax.random.normal(jax.random.PRNGKey(1), (2048, 32000))
labels = jax.random.randint(jax.random.PRNGKey(2), (2048,), 0, 32000)
def xent(z, y):
    return jnp.mean(jax.nn.logsumexp(z, axis=-1) - jnp.take_along_axis(z, y[:, None], axis=-1)[:, 0])
dump("ml_xent_vocab32k", xent, logits, labels)
dump("ml_xent_vocab32k_grad", jax.grad(xent), logits, labels)

# Llama-style block: RMSNorm + SwiGLU MLP, batch*seq 4096, d 1024, hidden 2816.
k = jax.random.split(jax.random.PRNGKey(3), 4)
x = jax.random.normal(k[0], (4096, 1024))
W = (jax.random.normal(k[1], (1024, 2816)) * 0.02, jax.random.normal(k[2], (1024, 2816)) * 0.02,
     jax.random.normal(k[3], (2816, 1024)) * 0.02, jnp.ones(1024))
def block(W, x):
    w1, w3, w2, g = W
    h = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * g
    return x + (jax.nn.silu(h @ w1) * (h @ w3)) @ w2
dump("ml_llama_mlp_fwd", block, W, x)
dump("ml_llama_mlp_grad", jax.grad(lambda W, x: jnp.sum(block(W, x) ** 2)), W, x)
