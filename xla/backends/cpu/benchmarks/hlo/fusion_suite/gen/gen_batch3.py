"""Third batch: ott-jax Sinkhorn, GP marginal likelihood, heat stencil, SPH."""
import os
import sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_lapack  # noqa: F401
import jax, jax.numpy as jnp
OUT = os.environ.get("OUT", ".") + "/"

def dump(name, f, *args):
    txt = jax.jit(f).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    cc = sorted(set(re.findall(r'custom_call_target="([^"]*)"', txt)))
    if cc: print("SKIP", name, cc); return
    open(OUT + name + ".hlo", "w").write(txt); print("wrote", name)

k = jax.random.split(jax.random.PRNGKey(0), 8)

# ott-jax: entropic OT between two 4096-point clouds in 3D, 50 iterations.
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
x = jax.random.normal(k[0], (4096, 3)); y = jax.random.normal(k[1], (4096, 3)) + 1.0
def ot_cost(x, y):
    geom = pointcloud.PointCloud(x, y, epsilon=0.05)
    out = sinkhorn.Sinkhorn(max_iterations=50, min_iterations=50, threshold=-1.0)(linear_problem.LinearProblem(geom))
    return out.reg_ot_cost
dump("ott_sinkhorn_4096", ot_cost, x, y)
dump("ott_sinkhorn_4096_grad", jax.grad(ot_cost), x, y)

# GP regression: negative log marginal likelihood, RBF kernel, N=2048, f32.
X = jax.random.normal(k[2], (2048, 4)); Y = jnp.sin(X).sum(-1)
def nlml(p, X, Y):
    ls, sf, sn = jnp.exp(p)
    d2 = jnp.sum((X[:, None] - X[None]) ** 2, -1)
    K = sf * jnp.exp(-0.5 * d2 / ls ** 2) + (sn + 1e-4) * jnp.eye(X.shape[0])
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), Y)
    return 0.5 * Y @ a + jnp.sum(jnp.log(jnp.diag(L)))
p0 = jnp.zeros(3)
dump("gp_nlml_2048", nlml, p0, X, Y)
dump("gp_nlml_2048_grad", jax.grad(nlml), p0, X, Y)

# 2D heat equation, 5-point stencil, 1024^2, 50 explicit steps.
u0 = jax.random.normal(k[3], (1024, 1024))
def heat(u):
    def step(_, u):
        lap = jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0) + jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1) - 4 * u
        return u + 0.2 * lap
    return jax.lax.fori_loop(0, 50, step, u)
dump("stencil_heat2d_1024", heat, u0)

# SPH: density and pressure force, 4096 particles, cubic spline, all pairs.
pos = jax.random.uniform(k[4], (4096, 3)) * 10.0
def sph(pos, h=0.5):
    r = pos[:, None] - pos[None]
    d = jnp.sqrt(jnp.sum(r * r, -1) + 1e-12)
    q = d / h
    w = jnp.where(q < 1, 1 - 1.5 * q**2 + 0.75 * q**3, jnp.where(q < 2, 0.25 * (2 - q) ** 3, 0.0))
    rho = jnp.sum(w, 1)
    p = 1.0 * (rho - 1.0)
    dw = jnp.where(q < 1, -3 * q + 2.25 * q**2, jnp.where(q < 2, -0.75 * (2 - q) ** 2, 0.0)) / h
    f = -(p[:, None] / rho[:, None] ** 2 + p[None] / rho[None] ** 2)[..., None] * dw[..., None] * r / d[..., None]
    return rho, jnp.sum(f, 1)
dump("sph_density_force_4096", sph, pos)
