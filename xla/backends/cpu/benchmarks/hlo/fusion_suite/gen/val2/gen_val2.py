"""Generate val2 held-out HLO benchmark modules. Run with .venv-val2."""
import os, sys, functools
sys.path.insert(0, os.path.expanduser("~/xla-bench/gen"))
import no_lapack  # noqa
import jax, jax.numpy as jnp, numpy as np
from jax import lax
OUT = os.path.expanduser("~/xla-bench/hlo/val2")
os.makedirs(OUT, exist_ok=True)
only = set(sys.argv[1:])

def emit(name, f, *args):
    if only and name not in only: return
    x64 = any(str(getattr(a, "dtype", "")) == "float64" for a in jax.tree_util.tree_leaves(args))
    jax.config.update("jax_enable_x64", x64)
    txt = jax.jit(f).lower(*args).compiler_ir("hlo").as_hlo_text()
    jax.config.update("jax_enable_x64", True)
    open(f"{OUT}/{name}.hlo", "w").write(txt)
    print(name, len(txt), "custom_call:", txt.count("custom_call_target"))

S = jax.ShapeDtypeStruct
f32, f64 = jnp.float32, jnp.float64
jax.config.update("jax_enable_x64", True)
def z(shape, dt=f32): return S(shape, dt)
def tree_spec(t): return jax.tree_util.tree_map(lambda a: S(a.shape, jnp.float32), t)

import haiku as hk
jax.config.update('jax_enable_x64', False)

# 1. haiku ResNet block (conv + groupnorm-ish layernorm), loss+grad
def resnet_fn(x):
    def block(h, c):
        r = h
        h = hk.Conv2D(c, 3, padding="SAME")(h); h = hk.GroupNorm(8)(h); h = jax.nn.relu(h)
        h = hk.Conv2D(c, 3, padding="SAME")(h); h = hk.GroupNorm(8)(h)
        return jax.nn.relu(h + r)
    h = hk.Conv2D(32, 3, padding="SAME")(x)
    for _ in range(3): h = block(h, 32)
    h = hk.avg_pool(h, 2, 2, "VALID")
    for _ in range(2): h = block(h, 32)
    return hk.Linear(10)(h.mean((1, 2)))
t = hk.without_apply_rng(hk.transform(resnet_fn))
xs = jnp.zeros((8, 32, 32, 3), f32)
p = t.init(jax.random.PRNGKey(0), xs)
def resnet_loss(p, x):
    return jnp.mean(jax.nn.logsumexp(t.apply(p, x), -1))
emit("haiku_resnet_grad_f32", jax.value_and_grad(resnet_loss), tree_spec(p), z(xs.shape))

# 2. haiku LSTM unrolled with scan (dynamic_unroll), loss+grad
def lstm_fn(x):
    core = hk.LSTM(128)
    out, _ = hk.dynamic_unroll(core, x, core.initial_state(x.shape[1]))
    return hk.Linear(16)(out)
t2 = hk.without_apply_rng(hk.transform(lstm_fn))
xs = jnp.zeros((64, 16, 32), f32)
p2 = t2.init(jax.random.PRNGKey(0), xs)
emit("haiku_lstm_grad_f32", jax.value_and_grad(lambda p, x: jnp.mean(t2.apply(p, x) ** 2)),
     tree_spec(p2), z(xs.shape))

# 3. haiku VAE loss + grad
def vae_fn(x, eps):
    h = jax.nn.gelu(hk.Linear(512)(x)); h = jax.nn.gelu(hk.Linear(256)(h))
    mu, lv = hk.Linear(32)(h), hk.Linear(32)(h)
    zz = mu + jnp.exp(0.5 * lv) * eps
    d = jax.nn.gelu(hk.Linear(256)(zz)); d = jax.nn.gelu(hk.Linear(512)(d))
    xr = hk.Linear(x.shape[-1])(d)
    rec = jnp.sum((xr - x) ** 2, -1)
    kl = -0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), -1)
    return jnp.mean(rec + kl)
t3 = hk.without_apply_rng(hk.transform(vae_fn))
x0, e0 = jnp.zeros((256, 784), f32), jnp.zeros((256, 32), f32)
p3 = t3.init(jax.random.PRNGKey(0), x0, e0)
emit("haiku_vae_grad_f32", jax.value_and_grad(t3.apply), tree_spec(p3), z(x0.shape), z(e0.shape))

# 4. haiku MLP-Mixer forward
def mixer_fn(x):
    B, P, C = x.shape
    h = hk.Linear(128)(x)
    for _ in range(4):
        y = hk.LayerNorm(-1, True, True)(h)
        y = jnp.swapaxes(y, 1, 2)
        y = hk.Linear(P)(jax.nn.gelu(hk.Linear(256)(y)))
        h = h + jnp.swapaxes(y, 1, 2)
        y = hk.LayerNorm(-1, True, True)(h)
        h = h + hk.Linear(128)(jax.nn.gelu(hk.Linear(512)(y)))
    return hk.Linear(10)(h.mean(1))
t4 = hk.without_apply_rng(hk.transform(mixer_fn))
x0 = jnp.zeros((32, 64, 48), f32)
p4 = t4.init(jax.random.PRNGKey(0), x0)
emit("haiku_mlpmixer_fwd_f32", t4.apply, tree_spec(p4), z(x0.shape))

jax.config.update('jax_enable_x64', True)
# 5. numpyro hierarchical model log_density + grad (f64)
import numpyro, numpyro.distributions as dist
from numpyro.infer.util import log_density
G, N = 64, 2048
def model(x, y):
    mu = numpyro.sample("mu", dist.Normal(0., 5.))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.))
    with numpyro.plate("g", G):
        a = numpyro.sample("a", dist.Normal(mu, tau))
        b = numpyro.sample("b", dist.Normal(0., 2.))
    sig = numpyro.sample("sig", dist.LogNormal(0., 1.))
    gid = jnp.arange(N) % G
    with numpyro.plate("n", N):
        numpyro.sample("y", dist.StudentT(4., a[gid] + b[gid] * x, sig), obs=y)
def np_lp(params, x, y):
    return log_density(model, (x, y), {}, params)[0]
params0 = dict(mu=z(()), tau=z(()), a=z((G,)), b=z((G,)), sig=z(()))
params0 = {k: S(v.shape, f64) for k, v in params0.items()}
emit("numpyro_hier_logdensity_grad_f64", jax.value_and_grad(np_lp), params0, z((N,), f64), z((N,), f64))

# 6. jax.scipy.signal 2D convolution + correlate
import jax.scipy.signal as jss, jax.scipy.ndimage as jsn, jax.scipy.special as jsp
def sig_fn(img, k):
    y = jss.convolve2d(img, k, mode="same")
    y2 = jss.correlate(img, k, mode="valid", method="direct")
    return y, y2
emit("jsp_signal_convolve2d_f32", sig_fn, z((512, 512)), z((9, 9)))

# 7. map_coordinates warp
def warp(img, d):
    H, W = img.shape
    yy, xx = jnp.meshgrid(jnp.arange(H, dtype=f32), jnp.arange(W, dtype=f32), indexing="ij")
    cy = yy + 4. * jnp.sin(xx / 17. + d[0]); cx = xx + 4. * jnp.cos(yy / 13. + d[1])
    o1 = jsn.map_coordinates(img, [cy, cx], order=1, mode="nearest")
    o2 = jsn.map_coordinates(o1, [cx.T, cy.T], order=1, mode="constant")
    return o2
emit("jsp_map_coordinates_f32", warp, z((512, 512)), z((2,)))

# 8. special functions, f64
def special(a, x):
    r = jsp.gammaln(a * 10) + jsp.digamma(a * 3) + jsp.erf(x) + jsp.erfc(x * .5) \
        + jsp.i0e(x * 4) + jsp.i1e(x * 4) + jsp.log_ndtr(x * 3 - 4) + jsp.xlogy(a, x) \
        + jsp.betaln(a, x) + jsp.gammainc(a * 4, x * 4) + jsp.polygamma(1, a * 2) + jsp.expit(x)
    return r.sum(), jnp.sort(r.ravel())[:8] if False else r
emit("jsp_special_f64", special, z((256, 1024), f64), z((256, 1024), f64))

# 9. sep-CMA-ES / ES generations, fori_loop, f64
PO, D = 256, 64
def rastrigin(x): return 10 * x.shape[-1] + jnp.sum(x ** 2 - 10 * jnp.cos(2 * jnp.pi * x), -1)
def es(m0, key_seed):
    key = jax.random.PRNGKey(0)
    mu = PO // 2
    w = jnp.log(mu + .5) - jnp.log(jnp.arange(1, mu + 1)); w = w / w.sum()
    mueff = 1 / jnp.sum(w ** 2)
    cs = (mueff + 2) / (D + mueff + 5); cc = 4 / (D + 4); c1 = 2 / ((D + 1.3) ** 2 + mueff) * (D + 2) / 3
    cmu = jnp.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff)) * (D + 2) / 3
    ds = 1 + cs + 2 * jnp.maximum(0, jnp.sqrt((mueff - 1) / (D + 1)) - 1)
    chiN = jnp.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
    def body(i, st):
        m, sg, C, ps, pc, k = st
        k, k1 = jax.random.split(k)
        zz = jax.random.normal(k1, (PO, D), m.dtype)
        y = jnp.sqrt(C) * zz
        x = m + sg * y
        f = rastrigin(x)
        idx = jnp.argsort(f)[:mu]
        ysel = y[idx]; zsel = zz[idx]
        yw = w @ ysel; zw = w @ zsel
        m = m + sg * yw
        ps = (1 - cs) * ps + jnp.sqrt(cs * (2 - cs) * mueff) * zw
        hs = (jnp.linalg.norm(ps) / chiN < 1.4 + 2 / (D + 1)).astype(m.dtype)
        pc = (1 - cc) * pc + hs * jnp.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * pc ** 2 + cmu * (w @ ysel ** 2)
        sg = sg * jnp.exp(cs / ds * (jnp.linalg.norm(ps) / chiN - 1))
        return m, sg, C, ps, pc, k
    st = (m0 * 3, jnp.asarray(1.0, m0.dtype), jnp.ones(D, m0.dtype), jnp.zeros(D, m0.dtype), jnp.zeros(D, m0.dtype), key)
    m, sg, C, *_ = lax.fori_loop(0, 50, body, st)
    return m, rastrigin(m), sg
emit("cmaes_sep_fori_f64", es, z((D,), f64), z((), f64))

# 10. 1D shallow water finite volume (Rusanov), fori_loop, f64
def swe(h0, u0):
    n = h0.shape[0]; dx = 1.0 / n; g = 9.81
    h = h0; hu = h0 * u0
    def flux(h, hu):
        u = hu / h
        return hu, hu * u + 0.5 * g * h * h
    def step(i, st):
        h, hu = st
        c = jnp.abs(hu / h) + jnp.sqrt(g * h)
        dt = 0.4 * dx / jnp.max(c)
        hl, hr = h, jnp.roll(h, -1); ml, mr = hu, jnp.roll(hu, -1)
        fl = flux(hl, ml); fr = flux(hr, mr)
        a = jnp.maximum(c, jnp.roll(c, -1))
        F0 = 0.5 * (fl[0] + fr[0]) - 0.5 * a * (hr - hl)
        F1 = 0.5 * (fl[1] + fr[1]) - 0.5 * a * (mr - ml)
        h = h - dt / dx * (F0 - jnp.roll(F0, 1))
        hu = hu - dt / dx * (F1 - jnp.roll(F1, 1))
        return h, hu
    h, hu = lax.fori_loop(0, 2000, step, (h, hu))
    return h, hu
emit("shallow_water_1d_fori_f64", swe, z((4096,), f64), z((4096,), f64))

# 11. PIC: scatter-add deposit + gather push, fori, f32
NP, NG = 200_000, 256
def pic(x0, v0):
    qm = -1.0; dx = 1.0 / NG; dt = 0.01
    x = x0 - 0.5 + 0.5 ; v = v0 - 1.0
    def step(i, st):
        x, v = st
        s = x * NG
        i0 = jnp.floor(s).astype(jnp.int32); fr = s - i0
        i0 = i0 % NG; i1 = (i0 + 1) % NG
        rho = jnp.zeros(NG, x.dtype).at[i0].add(1 - fr).at[i1].add(fr)
        rho = rho / (NP / NG) - 1.0
        k = jnp.fft.rfftfreq(NG, dx) * 2 * jnp.pi
        rk = jnp.fft.rfft(rho)
        E = jnp.fft.irfft(jnp.where(k > 0, -1j * rk / jnp.where(k > 0, k, 1), 0), NG).astype(x.dtype)
        Ep = E[i0] * (1 - fr) + E[i1] * fr
        v = v + qm * Ep * dt
        x = (x + v * dt) % 1.0
        return x, v
    x, v = lax.fori_loop(0, 50, step, (x, v))
    return x, v
emit("pic_scatter_fft_fori_f32", pic, z((NP,)), z((NP,)))

# 12. 2D wave equation with lax.conv, scan, f32
def wave(u0):
    k = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], f32)[None, None]
    c2 = 0.2
    def step(c, _):
        u, up = c
        lap = lax.conv_general_dilated(u[None, None], k, (1, 1), "SAME")[0, 0]
        un = 2 * u - up + c2 * lap
        un = un * (1 - 0.001)
        return (un, u), None
    (u, _), _ = lax.scan(step, (u0, u0), None, length=300)
    return u
emit("wave2d_conv_scan_f32", wave, z((512, 512)))

# 13. quantum circuit state vector, fori_loop over layers, complex64 (with grad)
NQ = 16
def qc(theta):
    st = jnp.zeros(2 ** NQ, jnp.complex64).at[0].set(1.0)
    idx = jnp.arange(2 ** NQ)
    def layer(l, st):
        th = theta[l]
        for q in range(NQ):
            a, b = th[q, 0], th[q, 1]
            c, s = jnp.cos(a / 2), jnp.sin(a / 2)
            U = jnp.array([[c, -s * jnp.exp(1j * b)], [s * jnp.exp(-1j * b), c]]).astype(jnp.complex64)
            v = st.reshape(2 ** q, 2, 2 ** (NQ - q - 1))
            st = jnp.einsum("ij,ajb->aib", U, v).reshape(-1)
        for q in range(NQ):
            bit = ((idx >> q) & 1) & ((idx >> ((q + 1) % NQ)) & 1)
            st = st * (1 - 2 * bit).astype(jnp.complex64)
        return st
    st = lax.fori_loop(0, theta.shape[0], layer, st)
    zz = 1 - 2 * ((idx & 1) ^ ((idx >> 1) & 1)).astype(f32)
    return jnp.sum(zz * (st.real ** 2 + st.imag ** 2))
emit("quantum_statevec_grad_c64", jax.grad(qc), z((8, NQ, 2)))

# 14. GMM EM, fori_loop, f32
def gmm(x, mu0):
    K = 16; N, D = x.shape
    mu = mu0[:K] + 0 * mu0[:K]; var = jnp.ones((K, D), x.dtype); pi = jnp.ones(K, x.dtype) / K
    def em(i, st):
        mu, var, pi = st
        ll = -0.5 * jnp.sum((x[:, None, :] - mu[None]) ** 2 / var[None] + jnp.log(2 * jnp.pi * var[None]), -1) + jnp.log(pi)
        r = jax.nn.softmax(ll, axis=1)
        nk = r.sum(0) + 1e-6
        mu = (r.T @ x) / nk[:, None]
        var = (r.T @ (x * x)) / nk[:, None] - mu ** 2 + 1e-3
        var = jnp.maximum(var, 1e-3)
        return mu, var, nk / N
    mu, var, pi = lax.fori_loop(0, 20, em, (mu, var, pi))
    return mu, var, pi
emit("gmm_em_fori_f32", gmm, z((8192, 16)), z((16, 16)))

# 15. k-means, fori_loop, f32 (segment_sum update)
def kmeans(x, c0):
    K = 32
    c = c0[:K] if c0.shape[0] >= K else c0
    def it(i, c):
        d = jnp.sum(x * x, 1)[:, None] - 2 * x @ c.T + jnp.sum(c * c, 1)[None]
        a = jnp.argmin(d, 1)
        s = jax.ops.segment_sum(x, a, K); n = jax.ops.segment_sum(jnp.ones(x.shape[0], x.dtype), a, K)
        return jnp.where(n[:, None] > 0, s / jnp.maximum(n, 1)[:, None], c)
    c = lax.fori_loop(0, 25, it, c)
    return c
emit("kmeans_fori_f32", kmeans, z((20000, 8)), z((32, 8)))

# 16. BCOO matvec power iteration, f32
from jax.experimental import sparse as jsparse
N16, NNZ = 20000, 200000
def power(vals, v0):
    i = jnp.arange(NNZ, dtype=jnp.int32)
    rows = (i * 7919) % N16; cols = (i * 104729 + i // 3) % N16
    A = jsparse.BCOO((vals, jnp.stack([rows, cols], 1)), shape=(N16, N16))
    def it(_, v):
        w = A @ v + A.T @ v
        return w / jnp.linalg.norm(w)
    v = lax.fori_loop(0, 100, it, v0 / jnp.linalg.norm(v0))
    return v, v @ (A @ v)
emit("bcoo_power_iter_f32", power, z((NNZ,)), z((N16,)))

# 17. Mandelbrot with while_loop (data-dependent, capped)
def mandel(s):
    H = W = 400
    yy, xx = jnp.meshgrid(jnp.arange(H, dtype=f32), jnp.arange(W, dtype=f32), indexing="ij")
    c = (xx / W * 3.0 - 2.0) * 1.0 + 1j * (yy / H * 2.4 - 1.2) * (s[0] / s[0])
    c = c.astype(jnp.complex64)
    def cond(st):
        i, zc, cnt = st
        return (i < 256) & jnp.any(jnp.abs(zc) <= 2.0)
    def body(st):
        i, zc, cnt = st
        act = jnp.abs(zc) <= 2.0
        zc = jnp.where(act, zc * zc + c, zc)
        return i + 1, zc, cnt + act.astype(jnp.int32)
    _, _, cnt = lax.while_loop(cond, body, (0, jnp.zeros_like(c), jnp.zeros((H, W), jnp.int32)))
    return cnt
emit("mandelbrot_while_c64", mandel, z((1,)))

# 18. ray-sphere renderer, f32
def render(cen, rad):
    H = W = 512; NS = 32
    ys, xs_ = jnp.meshgrid(jnp.linspace(-1, 1, H, dtype=f32), jnp.linspace(-1, 1, W, dtype=f32), indexing="ij")
    d = jnp.stack([xs_, ys, jnp.ones_like(xs_)], -1); d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    i = jnp.arange(NS, dtype=f32)
    C = jnp.stack([jnp.sin(i * 1.7) * 3 * cen[:NS, 0], jnp.cos(i * 2.3) * 2 * cen[:NS, 1], 6 + 3 * jnp.sin(i) * cen[:NS, 2]], -1)
    R = 0.3 * rad[:NS]
    oc = -C
    b = jnp.einsum("hwk,sk->hws", d, oc)
    cc = jnp.sum(oc * oc, -1) - R ** 2
    disc = b * b - cc
    t = -b - jnp.sqrt(jnp.maximum(disc, 0))
    t = jnp.where((disc > 0) & (t > 0), t, jnp.inf)
    ti = jnp.argmin(t, -1); tm = jnp.min(t, -1)
    hit = jnp.isfinite(tm)
    P = d * jnp.where(hit, tm, 0)[..., None]
    n = P - C[ti]; n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    L = jnp.array([0.5, 0.8, -0.4]); L = L / jnp.linalg.norm(L)
    diff = jnp.maximum(n @ L, 0)
    # shadow rays
    sd = jnp.broadcast_to(L, P.shape)
    oc2 = P[:, :, None, :] - C; b2 = jnp.sum(sd[:, :, None, :] * oc2, -1)
    c2 = jnp.sum(oc2 * oc2, -1) - R ** 2
    sh = jnp.any((b2 * b2 - c2 > 0) & (-b2 > 1e-3) & (jnp.arange(NS, dtype=jnp.int32) != ti[..., None]), -1)
    return jnp.where(hit, diff * jnp.where(sh, 0.3, 1.0), 0.0)
emit("ray_sphere_render_f32", render, z((32, 3)), z((32,)))

# 19. causal attention with grad, f32
def attn(q, k, v):
    B, H, T, Dh = q.shape
    def f(q, k, v):
        s = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(Dh)
        m = jnp.tril(jnp.ones((T, T), bool))
        s = jnp.where(m, s, -1e30)
        o = jnp.einsum("bhts,bhsd->bhtd", jax.nn.softmax(s, -1), v)
        return jnp.sum(o ** 2)
    return jax.grad(f, argnums=(0, 1, 2))(q, k, v)
emit("causal_attention_grad_f32", attn, z((4, 8, 256, 64)), z((4, 8, 256, 64)), z((4, 8, 256, 64)))

# 20. bilateral filter, f32
def bilateral(img):
    r = 5; ss, sr = 3.0, 0.2
    pad = jnp.pad(img, r, mode="reflect")
    H, W = img.shape
    num = jnp.zeros_like(img); den = jnp.zeros_like(img)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = pad[r + dy:r + dy + H, r + dx:r + dx + W]
            w = jnp.exp(-(dy * dy + dx * dx) / (2 * ss * ss) - (sh - img) ** 2 / (2 * sr * sr))
            num = num + w * sh; den = den + w
    return num / den
emit("bilateral_filter_f32", bilateral, z((768, 768)))

# 21. Monte Carlo option pricing with scan + grad, f64
def mc(s0, sig, r):
    NPATH, NSTEP = 100_000, 100
    key = jax.random.PRNGKey(0)
    dt = 1.0 / NSTEP
    def price(s0, sig, r):
        def step(c, k):
            S, m = c
            zz = jax.random.normal(k, (NPATH,), f64)
            S = S * jnp.exp((r - 0.5 * sig ** 2) * dt + sig * jnp.sqrt(dt) * zz)
            return (S, m + S), None
        (S, m), _ = lax.scan(step, (jnp.full(NPATH, s0), jnp.zeros(NPATH, f64)), jax.random.split(key, NSTEP))
        asian = jnp.maximum(m / NSTEP - 1.0 * s0, 0)
        return jnp.exp(-r) * asian.mean()
    return jax.value_and_grad(price, argnums=(0, 1, 2))(s0, sig, r)
emit("mc_asian_option_scan_grad_f64", mc, z((), f64), z((), f64), z((), f64))

# 22. ragged softmax via segment_max/segment_sum, f32
def ragged(x, w):
    N = x.shape[0]; NSEG = 4096
    seg = (jnp.arange(N, dtype=jnp.int32) * 40503 % (2 ** 20)) % NSEG
    seg = jnp.sort(seg)
    def f(x):
        m = jax.ops.segment_max(x, seg, NSEG)
        e = jnp.exp(x - m[seg])
        s = jax.ops.segment_sum(e, seg, NSEG)
        p = e / s[seg]
        return jnp.sum(p * w * jnp.log(p + 1e-9)), p
    (v, p), g = jax.value_and_grad(f, has_aux=True)(x)
    return v, p, g
emit("ragged_softmax_segment_grad_f32", ragged, z((1_000_000,)), z((1_000_000,)))
