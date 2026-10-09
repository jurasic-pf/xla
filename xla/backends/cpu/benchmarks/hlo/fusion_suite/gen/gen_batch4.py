"""Fourth batch: raytrax (ECRH ray tracing, W7-X), brax/MJX (robotics),
jaxlie (Lie groups, kinematics), flax (GPT block training, CNN).
Run with .venv-more. Writes hlo/libs/*.hlo; skips modules with custom calls."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
jax.config.update("jax_enable_x64", True)
import no_lapack  # noqa: F401
import jax.numpy as jnp
import numpy as np

OUT = os.environ.get("OUT", ".") + "/"
WHICH = set(sys.argv[1:]) or {"raytrax", "brax", "jaxlie", "flax"}

def dump(name, f, *args):
    txt = jax.jit(f).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    cc = sorted(set(re.findall(r'custom_call_target="([^"]*)"', txt)))
    if cc:
        print("SKIP", name, cc); return
    open(OUT + name + ".hlo", "w").write(txt); print("wrote", name, len(txt) // 1024, "KiB")

if "raytrax" in WHICH:
    import raytrax
    from raytrax.examples.w7x import get_w7x_magnetic_configuration
    conf = get_w7x_magnetic_configuration()
    rho = jnp.linspace(0, 1, 200)
    prof = raytrax.RadialProfiles(rho=rho, electron_density=2.0 * (1 - rho**2),
                                  electron_temperature=3.0 * (1 - rho**2))
    def run(direction):
        beam = raytrax.Beam(position=jnp.array([6.6, 0.0, 0.0]), direction=direction,
                            frequency=140e9, mode="O", power=1e6)
        r = raytrax.trace(conf, prof, beam, trim=False)
        return r.optical_depth
    d0 = jnp.array([-0.985, 0.0, -0.174])
    dump("raytrax_w7x_trace", run, d0)
    dump("raytrax_w7x_trace_grad", jax.grad(run), d0)

if "brax" in WHICH:
    jax.config.update("jax_enable_x64", False)
    from brax import envs
    env = envs.get_environment("ant", backend="mjx")
    keys = jax.random.split(jax.random.PRNGKey(0), 1024)
    state = jax.jit(jax.vmap(env.reset))(keys)
    act = jnp.zeros((1024, env.action_size))
    dump("brax_mjx_ant_step_1024", jax.vmap(env.step), state, act)
    def rollout(state, act):
        def body(s, _):
            s = jax.vmap(env.step)(s, act); return s, s.reward
        return jax.lax.scan(body, state, None, length=16)[1]
    dump("brax_mjx_ant_rollout16_1024", rollout, state, act)

if "jaxlie" in WHICH:
    import jaxlie
    jax.config.update("jax_enable_x64", False)
    k = jax.random.split(jax.random.PRNGKey(1), 3)
    a = jaxlie.SE3.exp(jax.random.normal(k[0], (1 << 20, 6)) * 0.3)
    b = jaxlie.SE3.exp(jax.random.normal(k[1], (1 << 20, 6)) * 0.3)
    dump("jaxlie_se3_compose_log_1M", lambda a, b: (a @ b.inverse()).log(), a, b)
    # 16-joint serial chain, 65536 configurations: end-effector pose and its gradient.
    def fk(q):
        T = jaxlie.SE3.identity(batch_axes=q.shape[:1])
        for j in range(q.shape[1]):
            T = T @ jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_z_radians(q[:, j]), jnp.array([0.1, 0.0, 0.0])) @ \
                jaxlie.SE3.from_rotation(jaxlie.SO3.from_x_radians(0.3 * jnp.ones_like(q[:, j])))
        return T.translation()
    q = jax.random.normal(k[2], (65536, 16))
    dump("jaxlie_fk16_65536", fk, q)
    dump("jaxlie_fk16_65536_grad", jax.grad(lambda q: jnp.sum(fk(q) ** 2)), q)

if "flax" in WHICH:
    jax.config.update("jax_enable_x64", False)
    import flax.linen as nn
    class Block(nn.Module):
        d: int
        @nn.compact
        def __call__(self, x):
            h = nn.LayerNorm()(x)
            h = nn.SelfAttention(num_heads=8, qkv_features=self.d)(h, mask=nn.make_causal_mask(x[..., 0]))
            x = x + h
            h = nn.Dense(4 * self.d)(nn.LayerNorm()(x))
            return x + nn.Dense(self.d)(nn.gelu(h))
    class GPT(nn.Module):
        @nn.compact
        def __call__(self, tok):
            x = nn.Embed(8192, 256)(tok)
            for _ in range(4):
                x = Block(256)(x)
            return nn.Dense(8192)(nn.LayerNorm()(x))
    gpt = GPT(); tok = jnp.zeros((8, 256), jnp.int32)
    params = gpt.init(jax.random.PRNGKey(0), tok)
    def loss(p, tok):
        logits = gpt.apply(p, tok[:, :-1])
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), tok[:, 1:, None], -1))
    dump("flax_gpt4_train_grad", jax.grad(loss), params, tok)
    class CNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            for c in (32, 64, 128):
                r = x
                x = nn.relu(nn.BatchNorm(use_running_average=True)(nn.Conv(c, (3, 3))(x)))
                x = nn.Conv(c, (3, 3))(x)
                x = nn.relu(x + nn.Conv(c, (1, 1))(r))
                x = nn.max_pool(x, (2, 2), (2, 2))
            return nn.Dense(10)(x.mean((1, 2)))
    cnn = CNN(); xi = jnp.zeros((32, 64, 64, 3))
    v = cnn.init(jax.random.PRNGKey(0), xi)
    dump("flax_resnetish_fwd", lambda v, x: cnn.apply(v, x), v, xi)
