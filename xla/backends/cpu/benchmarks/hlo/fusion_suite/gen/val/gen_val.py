"""Shared generator for held-out validation HLO modules. Usage: gen_val.py [name ...]  (default: all)
Run with ~/xla-bench/.venv-val/bin/python. Writes ~/xla-bench/hlo/val/<name>.hlo (unoptimized HLO text)."""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import numpy as np
OUT = os.path.expanduser("~/xla-bench/hlo/val")
S = jax.ShapeDtypeStruct
f32, f64 = jnp.float32, jnp.float64
REG = {}
def mod(name):
    def d(fn): REG[name] = fn; return fn
    return d

@mod("optax_adamw_f32")
def _():
    import optax
    shapes = {"w%d" % i: (1024, 1024) for i in range(6)} | {"emb": (8192, 256), "b": (4096,)}
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=1e-2))
    def f(params, grads, mu, nu):
        clip, (adam, *rest) = tx.init(params)
        adam = adam._replace(mu=mu, nu=nu, count=jnp.asarray(10, jnp.int32))
        upd, _ = tx.update(grads, (clip, (adam, *rest)), params)
        return optax.apply_updates(params, upd)
    t = {k: S(v, f32) for k, v in shapes.items()}
    return f, (t, t, t, t)

@mod("optax_lion_scan_grad_f32")
def _():
    import optax
    tx = optax.lion(1e-3)
    def f(w, X, y):
        def loss(w): return jnp.mean((jnp.tanh(X @ w) - y) ** 2)
        def step(c, _):
            w, st = c
            g = jax.grad(loss)(w)
            u, st = tx.update(g, st, w)
            return (optax.apply_updates(w, u), st), None
        (w, _), _ = lax.scan(step, (w, tx.init(w)), None, length=30)
        return w, loss(w)
    return f, (S((256, 256), f32), S((4096, 256), f32), S((4096, 256), f32))

def _logreg(d, n):
    def logp(w, X, y):
        z = X @ w
        return jnp.sum(y * z - jnp.logaddexp(0.0, z)) - 0.5 * jnp.sum(w ** 2)
    return logp

@mod("blackjax_hmc_f64")
def _():
    import blackjax
    logp = _logreg(0, 0)
    def f(w0, X, y):
        hmc = blackjax.hmc(lambda w: logp(w, X, y), step_size=1e-2, inverse_mass_matrix=jnp.ones(w0.shape[0]), num_integration_steps=25)
        st = hmc.init(w0)
        def one(st, k):
            st, info = hmc.step(k, st); return st, info.acceptance_rate
        st, acc = lax.scan(one, st, jax.random.split(jax.random.PRNGKey(0), 4))
        return st.position, acc
    return f, (S((64,), f64), S((4096, 64), f64), S((4096,), f64))

@mod("blackjax_nuts_f64")
def _():
    import blackjax
    logp = _logreg(0, 0)
    def f(w0, X, y):
        nuts = blackjax.nuts(lambda w: logp(w, X, y), step_size=5e-3, inverse_mass_matrix=jnp.ones(w0.shape[0]), max_num_doublings=6)
        st = nuts.init(w0)
        st, info = nuts.step(jax.random.PRNGKey(1), st)
        return st.position, info.num_integration_steps
    return f, (S((32,), f64), S((2048, 32), f64), S((2048,), f64))

@mod("optimistix_lbfgs_f64")
def _():
    import optimistix as optx
    def f(x0, a):
        def fn(x, args):
            a = args
            return jnp.sum(a[:-1] * 100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        sol = optx.minimise(fn, optx.BFGS(rtol=1e-8, atol=1e-8), x0, args=a, max_steps=40, throw=False)
        return sol.value, sol.stats["num_steps"]
    return f, (S((256,), f64), S((256,), f64))

@mod("lineax_cg_f64")
def _():
    import lineax as lx
    def f(diag, b):
        n = b.shape[0]
        def mv(x):
            xl = jnp.concatenate([x[:1] * 0, x[:-1]]); xr = jnp.concatenate([x[1:], x[:1] * 0])
            return (2.0 + diag) * x - xl - xr
        op = lx.FunctionLinearOperator(mv, jax.ShapeDtypeStruct((n,), f64), lx.positive_semidefinite_tag)
        sol = lx.linear_solve(op, b, lx.CG(rtol=1e-10, atol=1e-10, max_steps=300), throw=False)
        return sol.value
    return f, (S((1 << 20,), f64), S((1 << 20,), f64))

@mod("lineax_gmres_f32")
def _():
    import lineax as lx
    def f(A, b):
        n = b.shape[0]
        A = A / n + 4.0 * jnp.eye(n, dtype=A.dtype)
        sol = lx.linear_solve(lx.MatrixLinearOperator(A), b, lx.GMRES(rtol=1e-5, atol=1e-5, restart=30, max_steps=60), throw=False)
        return sol.value
    return f, (S((2048, 2048), f32), S((2048,), f32))

@mod("distrax_gmm_logprob_grad_f32")
def _():
    import distrax
    def f(logits, mu, sc, x):
        def nll(p):
            logits, mu, sc = p
            d = distrax.MixtureSameFamily(distrax.Categorical(logits=logits),
                distrax.MultivariateNormalDiag(mu, jax.nn.softplus(sc)))
            return -jnp.mean(d.log_prob(x))
        return jax.value_and_grad(nll)((logits, mu, sc))
    return f, (S((32,), f32), S((32, 16), f32), S((32, 16), f32), S((65536, 16), f32))

@mod("neural_ode_rk4_grad_f32")
def _():
    def f(W1, W2, x0, target):
        def vf(x): return jnp.tanh(x @ W1) @ W2
        def loss(W1, W2):
            def vf(x): return jnp.tanh(x @ W1) @ W2
            def step(x, _):
                h = 0.05
                k1 = vf(x); k2 = vf(x + h / 2 * k1); k3 = vf(x + h / 2 * k2); k4 = vf(x + h * k3)
                return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None
            x, _ = lax.scan(step, x0, None, length=40)
            return jnp.mean((x - target) ** 2)
        return jax.grad(loss, argnums=(0, 1))(W1, W2)
    return f, (S((128, 512), f32), S((512, 128), f32), S((512, 128), f32), S((512, 128), f32))

@mod("mps_contraction_f64")
def _():
    L, D, d = 24, 64, 2
    def f(A, H):
        # A: (L,D,d,D) MPS tensors, H: (d,d) local operator; <psi|H_i|psi> for all sites via transfer matrices
        A = A / jnp.sqrt(D * d)
        def tm(E, a, Op):
            t = jnp.einsum("xy,xsa->ysa", E, a)
            t = jnp.einsum("st,ysa->yta", Op, t)
            return jnp.einsum("yta,ytb->ab", t, a)
        I = jnp.eye(d)
        def step(E, a): return tm(E, a, I), None
        def site(i):
            def body(j, E): return tm(E, A[j], jnp.where(j == i, H, I))
            return lax.fori_loop(0, L, body, jnp.eye(D))
        norm, _ = lax.scan(step, jnp.eye(D), A)
        ex = jax.vmap(lambda i: jnp.trace(site(i)))(jnp.arange(0, L, 4))
        return jnp.trace(norm), ex
    return f, (S((24, 64, 2, 64), f64), S((2, 2), f64))

@mod("dynamax_kalman_f64")
def _():
    from dynamax.linear_gaussian_ssm import lgssm_filter, ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions
    ds, de, T = 16, 8, 2000
    def f(A, Hm, ys):
        A = A / (jnp.linalg.norm(A, ord=2) * 1.05) if False else A / (jnp.sqrt(jnp.sum(A * A)) * 1.1)
        p = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=jnp.zeros(ds), cov=jnp.eye(ds)),
            dynamics=ParamsLGSSMDynamics(weights=A, bias=jnp.zeros(ds), input_weights=jnp.zeros((ds, 0)), cov=0.1 * jnp.eye(ds)),
            emissions=ParamsLGSSMEmissions(weights=Hm, bias=jnp.zeros(de), input_weights=jnp.zeros((de, 0)), cov=0.5 * jnp.eye(de)))
        out = lgssm_filter(p, ys)
        return out.marginal_loglik, out.filtered_means
    return f, (S((ds, ds), f64), S((de, ds), f64), S((T, de), f64))

@mod("lorenz96_ensemble_f64")
def _():
    N, T = 40, 400
    def f(x0, noise):
        def rhs(x): return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + 8.0
        def run(x, nz):
            def step(x, _):
                h = 0.01
                k1 = rhs(x); k2 = rhs(x + h / 2 * k1); k3 = rhs(x + h / 2 * k2); k4 = rhs(x + h * k3)
                return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None
            x, _ = lax.scan(step, x + 1e-3 * nz, None, length=T)
            return x
        return jax.vmap(run, in_axes=(None, 0))(x0, noise)
    return f, (S((N,), f64), S((8192, N), f64))

@mod("game_of_life_f32")
def _():
    def f(x):
        b = (x > 1.0).astype(jnp.int8)
        def step(_, b):
            n = sum(jnp.roll(jnp.roll(b, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
            return ((n == 3) | ((b == 1) & (n == 2))).astype(jnp.int8)
        b = lax.fori_loop(0, 60, step, b)
        return b.sum(), b
    return f, (S((1024, 1024), f32),)

@mod("spectral_burgers_f64")
def _():
    N, T = 2048, 150
    def f(u0):
        k = jnp.fft.rfftfreq(N, 1.0 / N)
        mask = (k < N / 3).astype(f64); nu = 0.01
        def rhs(uh):
            u = jnp.fft.irfft(uh, n=N, axis=-1)
            return -0.5j * k * jnp.fft.rfft(u * u, axis=-1) * mask - nu * k * k * uh
        def step(uh, _):
            h = 1e-4
            k1 = rhs(uh); k2 = rhs(uh + h / 2 * k1); k3 = rhs(uh + h / 2 * k2); k4 = rhs(uh + h * k3)
            return uh + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None
        uh, _ = lax.scan(step, jnp.fft.rfft(u0 - 1.0, axis=-1), None, length=T)
        return jnp.fft.irfft(uh, n=N, axis=-1)
    return f, (S((128, N), f64),)

@mod("topk_sort_f32")
def _():
    def f(x, q):
        # lax.top_k lowers to a topk op the jaxlib 0.6.2 parser rejects (is_stable); use sort-based top-k
        nv, i = lax.sort_key_val(-x, jnp.broadcast_to(jnp.arange(x.shape[1], dtype=jnp.int32), x.shape), dimension=-1)
        v, i = -nv[:, :64], i[:, :64]
        s = jnp.sort(x, axis=-1)
        a = jnp.argsort(x[:, :4096], axis=-1)
        c = jnp.cumsum(s, axis=-1)
        idx = jax.vmap(jnp.searchsorted)(s, q)
        med = s[:, s.shape[1] // 2]
        return v.sum(-1) + i.sum(-1).astype(f32) + a[:, 0] + c[:, -1] + idx.sum(-1) + med
    return f, (S((1024, 8192), f32), S((1024, 256), f32))

@mod("jraph_gnn_grad_f32")
def _():
    import jraph
    Nn, E, H = 20000, 120000, 64
    def f(Wn, We, Wg, nodes, edges):
        s = (jnp.arange(E) * 7919) % Nn; r = (jnp.arange(E) * 104729 + 13) % Nn
        def loss(Wn, We, Wg):
            g = jraph.GraphsTuple(nodes=nodes, edges=edges, senders=s, receivers=r,
                                  n_node=jnp.array([Nn]), n_edge=jnp.array([E]), globals=None)
            for _ in range(3):
                net = jraph.GraphNetwork(
                    update_edge_fn=lambda e, sn, rn, gl: jax.nn.relu(jnp.concatenate([e, sn, rn], -1) @ We),
                    update_node_fn=lambda n, se, re, gl: jax.nn.relu(jnp.concatenate([n, re], -1) @ Wn),
                    aggregate_edges_for_nodes_fn=jraph.segment_sum)
                g = net(g)
            return jnp.mean((g.nodes @ Wg) ** 2)
        return jax.grad(loss, argnums=(0, 1, 2))(Wn, We, Wg)
    return f, (S((2 * H, H), f32), S((3 * H, H), f32), S((H, 1), f32), S((Nn, H), f32), S((E, H), f32))

@mod("transformer_decode_kv_f32")
def _():
    L, d, nh, Tm, ff, B = 8, 512, 8, 2048, 2048, 8
    hd = d // nh
    def f(x, pos, Wqkv, Wo, W1, W2, kc, vc):
        sc = 1.0 / jnp.sqrt(d)
        z = jnp.int32(0)
        def layer(x, p):
            wqkv, wo, w1, w2, k, v = p
            h = x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
            qkv = (h @ (wqkv * sc)).reshape(B, 3, nh, hd)
            q, kn, vn = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            k = lax.dynamic_update_slice(k, kn[:, :, None, :], (z, z, pos, z))
            v = lax.dynamic_update_slice(v, vn[:, :, None, :], (z, z, pos, z))
            att = jnp.einsum("bhd,bhtd->bht", q, k) / jnp.sqrt(hd)
            att = jnp.where(jnp.arange(Tm) <= pos, att, -1e9)
            att = jax.nn.softmax(att, -1)
            o = jnp.einsum("bht,bhtd->bhd", att, v).reshape(B, d)
            x = x + o @ (wo * sc)
            h = x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
            x = x + jax.nn.gelu(h @ (w1 * sc)) @ (w2 * (1.0 / jnp.sqrt(ff)))
            return x, (k, v)
        x, (kc2, vc2) = lax.scan(layer, x, (Wqkv, Wo, W1, W2, kc, vc))
        return x, kc2, vc2
    return f, (S((B, d), f32), S((), jnp.int32), S((L, d, 3 * d), f32), S((L, d, d), f32),
               S((L, d, ff), f32), S((L, ff, d), f32), S((L, B, nh, Tm, hd), f32), S((L, B, nh, Tm, hd), f32))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in (sys.argv[1:] or list(REG)):
        try:
            f, specs = REG[n]()
            low = jax.jit(f).lower(*specs)
            try: txt = low.compiler_ir("hlo").as_hlo_text()
            except Exception: txt = low.as_text(dialect="hlo")
            open(f"{OUT}/{n}.hlo", "w").write(txt)
            print("gen ok", n, len(txt), flush=True)
        except Exception as e:
            print("gen FAIL", n, type(e).__name__, str(e)[:300], flush=True)
