"""Equinox models: forward and one Adam training step. Writes hlo/libs/eqx_*.hlo."""
import os
import jax, jax.numpy as jnp, equinox as eqx, optax

OUT = os.environ.get("OUT", ".") + "/"
key = jax.random.PRNGKey(0)

def dump(name, f, *args):
    arrays, static = eqx.partition(args, eqx.is_array)
    g = lambda a: f(*eqx.combine(a, static))
    open(OUT + name + ".hlo", "w").write(jax.jit(g).lower(arrays).compiler_ir(dialect="hlo").as_hlo_text())
    print("wrote", name)

def train_step_fn(model, loss):
    opt = optax.adam(1e-3)
    params, static = eqx.partition(model, eqx.is_array)
    state = opt.init(params)
    def step(params, state, x, y):
        l, g = jax.value_and_grad(lambda p: loss(eqx.combine(p, static), x, y))(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l
    return step, params, state

# MLP: batch 4096, width 1024, GELU.
mlp = eqx.nn.MLP(256, 10, 1024, 4, activation=jax.nn.gelu, key=key)
x = jax.random.normal(key, (4096, 256)); y = jax.random.randint(key, (4096,), 0, 10)
def mlp_loss(m, x, y):
    logits = jax.vmap(m)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
dump("eqx_mlp_fwd", lambda m, x: jax.vmap(m)(x), mlp, x)
step, p, s = train_step_fn(mlp, mlp_loss); dump("eqx_mlp_train", step, p, s, x, y)

# Transformer block: batch 8, seq 512, dim 256, 8 heads.
class Block(eqx.Module):
    ln1: eqx.nn.LayerNorm; att: eqx.nn.MultiheadAttention; ln2: eqx.nn.LayerNorm; mlp: eqx.nn.MLP
    def __init__(self, d, h, key):
        k1, k2 = jax.random.split(key)
        self.ln1 = eqx.nn.LayerNorm(d); self.ln2 = eqx.nn.LayerNorm(d)
        self.att = eqx.nn.MultiheadAttention(h, d, key=k1)
        self.mlp = eqx.nn.MLP(d, d, 4 * d, 1, activation=jax.nn.gelu, key=k2)
    def __call__(self, x):
        h = jax.vmap(self.ln1)(x); x = x + self.att(h, h, h)
        return x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
blk = Block(256, 8, key)
xs = jax.random.normal(key, (8, 512, 256))
dump("eqx_transformer_fwd", lambda m, x: jax.vmap(m)(x), blk, xs)
step, p, s = train_step_fn(blk, lambda m, x, y: jnp.mean((jax.vmap(m)(x) - y) ** 2))
dump("eqx_transformer_train", step, p, s, xs, xs)

# CNN: batch 64, 3x64x64 images.
class CNN(eqx.Module):
    c1: eqx.nn.Conv2d; c2: eqx.nn.Conv2d; lin: eqx.nn.Linear
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.c1 = eqx.nn.Conv2d(3, 32, 3, padding=1, key=k1); self.c2 = eqx.nn.Conv2d(32, 64, 3, padding=1, key=k2)
        self.lin = eqx.nn.Linear(64, 10, key=k3)
    def __call__(self, x):
        x = jax.nn.relu(self.c1(x)); x = jax.nn.relu(self.c2(x))
        return self.lin(jnp.mean(x, axis=(1, 2)))
cnn = CNN(key); xi = jax.random.normal(key, (64, 3, 64, 64)); yi = jax.random.randint(key, (64,), 0, 10)
dump("eqx_cnn_fwd", lambda m, x: jax.vmap(m)(x), cnn, xi)
step, p, s = train_step_fn(cnn, mlp_loss); dump("eqx_cnn_train", step, p, s, xi, yi)
