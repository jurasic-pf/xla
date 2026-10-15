"""Fifth batch: exponax (ETDRK spectral PDE solvers) and jraph (GNN message passing).
Usage (.venv-more, exponax 0.2.0, jraph 0.0.6.dev0): gen_batch5.py [exponax] [jraph]
Writes hlo/libs/*.hlo; skips modules with custom calls."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import no_lapack  # noqa: F401
import jax.numpy as jnp
import numpy as np

OUT = os.environ.get("OUT", ".") + "/"
WHICH = set(sys.argv[1:]) or {"exponax", "jraph"}

def dump(name, f, *args):
    txt = jax.jit(f).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    cc = sorted(set(re.findall(r'custom_call_target="([^"]*)"', txt)))
    if cc:
        print("SKIP", name, cc); return
    open(OUT + name + ".hlo", "w").write(txt); print("wrote", name, len(txt) // 1024, "KiB")

if "exponax" in WHICH:
    import exponax as ex
    N, STEPS = 256, 100
    # 2D Kuramoto-Sivashinsky, ETDRK2 (stepper default order), 256x256, f32.
    stepper = ex.stepper.KuramotoSivashinsky(2, 60.0, N, 0.5)
    u0 = ex.ic.RandomTruncatedFourierSeries(2, cutoff=5)(N, key=jax.random.PRNGKey(0))
    target = jax.random.normal(jax.random.PRNGKey(1), u0.shape)
    def rollout(u0):
        def body(u, _):
            u = stepper(u); return u, None
        return jax.lax.scan(body, u0, None, length=STEPS)[0]
    def loss(u0):
        return jnp.mean((rollout(u0) - target) ** 2)
    dump("exponax_ks2d_rollout", rollout, u0)
    dump("exponax_ks2d_rollout_grad", jax.grad(loss), u0)

if "jraph" in WHICH:
    import jraph
    NN, NE, H, L = 50_000, 500_000, 64, 2
    k = jax.random.split(jax.random.PRNGKey(0), 8)
    senders = jax.random.randint(k[0], (NE,), 0, NN)
    receivers = jax.random.randint(k[1], (NE,), 0, NN)
    nodes = jax.random.normal(k[2], (NN, H))
    edges = jax.random.normal(k[3], (NE, H))
    graph = jraph.GraphsTuple(nodes=nodes, edges=edges, senders=senders, receivers=receivers,
                              globals=jnp.zeros((1, H)), n_node=jnp.array([NN]), n_edge=jnp.array([NE]))
    def mlp_init(key, sizes):
        ks = jax.random.split(key, len(sizes) - 1)
        return [(jax.random.normal(kk, (a, b)) / np.sqrt(a), jnp.zeros(b))
                for kk, a, b in zip(ks, sizes[:-1], sizes[1:])]
    def mlp(p, x):
        for i, (w, b) in enumerate(p):
            x = x @ w + b
            if i < len(p) - 1: x = jax.nn.relu(x)
        return x
    def ln(x):
        m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) * jax.lax.rsqrt(v + 1e-5)
    params = []
    for i in range(L):
        kk = jax.random.split(k[4 + i % 4], 2)
        params.append({"edge": mlp_init(kk[0], [3 * H, H, H]), "node": mlp_init(kk[1], [2 * H, H, H])})
    # jraph.GraphNetwork (interaction network: edge MLP on [e, x_s, x_r], node MLP on [x, sum_in e]),
    # residual + layer norm, 2 layers, hidden 64.
    def model(params, graph):
        for p in params:
            net = jraph.GraphNetwork(
                update_edge_fn=lambda e, s, r, g, p=p: e + ln(mlp(p["edge"], jnp.concatenate([e, s, r], -1))),
                update_node_fn=lambda n, se, re_, g, p=p: n + ln(mlp(p["node"], jnp.concatenate([n, re_], -1))),
                aggregate_edges_for_nodes_fn=jraph.segment_sum)
            graph = net(graph)
        return graph
    def fwd(params, graph):
        return model(params, graph).nodes
    def loss(params, graph):
        return jnp.mean(model(params, graph).nodes ** 2)
    dump("jraph_gn2_fwd", fwd, params, graph)
    dump("jraph_gn2_grad", jax.grad(loss), params, graph)
