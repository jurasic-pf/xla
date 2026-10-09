# Held-out validation set (do not inspect for tuning)
Env: `~/xla-bench/.venv-val` (jax 0.11.2 CPU, optax 0.2.8, blackjax 1.7.1, optimistix 0.1.0, lineax 0.1.1, distrax 0.1.9, dynamax 1.0.2, jraph 0.0.6.dev0).
Generator: `~/xla-bench/gen/val/gen_val.py [name...]` (shared; `gen_<name>.py` are thin wrappers). All lowered with jax_enable_x64 on, from ShapeDtypeStructs. Unoptimized HLO.
Data-dependent while loops are marked WHILE. All passed `jaxver/prep.py`. Runtimes are NOT measured (sized by estimate).

| name | library | computes | inputs |
|---|---|---|---|
| optax_adamw_f32 | optax | clip+AdamW update on 8-leaf param pytree | 4 pytrees (params,grads,mu,nu): 6x[1024,1024], [8192,256], [4096] f32 |
| optax_lion_scan_grad_f32 | optax | 30 Lion steps (scan) on tanh regression, grad inside | w[256,256], X[4096,256], y[4096,256] f32 |
| blackjax_hmc_f64 | blackjax | 4 HMC steps x 25 leapfrog, logistic regression | w[64], X[4096,64], y[4096] f64 |
| blackjax_nuts_f64 | blackjax | one NUTS step (max 6 doublings), logistic regression; WHILE | w[32], X[2048,32], y[2048] f64 |
| optimistix_lbfgs_f64 | optimistix | BFGS minimise of Rosenbrock-type fn, max 40 steps; WHILE | x0[256], a[256] f64 |
| lineax_cg_f64 | lineax | matrix-free CG on tridiagonal SPD op, max 300 it; WHILE | diag[2^20], b[2^20] f64 |
| lineax_gmres_f32 | lineax | restarted GMRES on dense diag-dominant system; WHILE | A[2048,2048], b[2048] f32 |
| distrax_gmm_logprob_grad_f32 | distrax | value+grad of Gaussian-mixture NLL | logits[32], mu[32,16], sc[32,16], x[65536,16] f32 |
| neural_ode_rk4_grad_f32 | plain JAX | grad of loss through 40 RK4 steps (scan) of MLP vector field | W1[128,512], W2[512,128], x0,target[512,128] f32 |
| mps_contraction_f64 | plain JAX einsum | MPS norm + local expectation values via transfer matrices (scan, fori, vmap) | A[24,64,2,64], H[2,2] f64 |
| dynamax_kalman_f64 | dynamax | LGSSM Kalman filter (scan) | A[16,16], H[8,16], ys[2000,8] f64 |
| lorenz96_ensemble_f64 | plain JAX | vmap over 8192 members of 400 RK4 steps (scan) | x0[40], noise[8192,40] f64 |
| game_of_life_f32 | plain JAX | Conway CA, 60 steps fori_loop, int8 board | x[1024,1024] f32 (thresholded) |
| spectral_burgers_f64 | plain JAX FFT | pseudo-spectral Burgers, 150 RK4 steps (scan), batch 128 | u0[128,2048] f64 |
| topk_sort_f32 | plain JAX | sort / key-value sort top-64 / argsort / cumsum / searchsorted (no loop) | x[1024,8192], q[1024,256] f32 |
| jraph_gnn_grad_f32 | jraph | grad of 3-layer GraphNetwork (segment_sum), indices built from iota | Wn[128,64], We[192,64], Wg[64,1], nodes[20000,64], edges[120000,64] f32 |
| transformer_decode_kv_f32 | plain JAX | one decode step, 8 layers (scan), KV cache update, B=8, T=2048 | x[8,512] f32, pos i32 scalar, weights stacked over L=8, kc,vc[8,8,8,2048,64] f32 |

Notes: lax.top_k was avoided (jaxlib 0.6.2 parser rejects `topk ... is_stable`). blackjax is f64 because x64 promotion made f32 carries inconsistent. Not included: pgx, jax-cfd, jax-cosmo (not installed/attempted).
