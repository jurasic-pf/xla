# Library benchmark modules

Pre-optimization HLO from real JAX library code, used as regression tests
for the XLA:CPU fusion changes. All modules are free of custom calls:
`gen/no_lapack.py` registers JAX's pure-HLO lowerings for LU, triangular
solve and Cholesky on CPU before tracing.

| file | source | how |
|---|---|---|
| `eqx_{mlp,transformer,cnn}_{fwd,train}` | equinox 0.13.8, optax | `gen/gen_equinox.py`: MLP 256-1024x4-10 on batch 4096 (GELU); transformer block d=256, 8 heads, seq 512, batch 8; CNN 2 conv + linear on 64x3x64x64. `train` = one Adam step. |
| `torax_{iterhybrid,basic}_step` | torax 1.4.3 | `gen/gen_torax.py <example>`: the jitted step function (`jit___call`) dumped while running `examples/iterhybrid_rampup.py` and `examples/basic_config.py`. |
| `desc_{jvp_scaled_error,compute_scaled_error,jac_scaled,zernike_radial}` | DESC 0.17.3 (jax 0.9.2, `.venv-desc`) | `gen/gen_desc.py`: HELIOTRON at L=M=8, N=3, two solver iterations. |
| `jaxfem_hyper_kernel_jac`, `jaxfem_solver_while` | jax-fem | `gen/gen_jaxfem.py 24`: Neo-Hookean hyperelasticity, 24^3 HEX8 box; element Jacobian kernel and the iterative linear solver loop. |
| `diffrax_robertson_kvaerno5` | diffrax 0.7.2 | `gen/gen_batch2.py` (with `EQX_ON_ERROR=nan` to drop error callbacks): stiff Robertson kinetics, Kvaerno5 + PID, vmapped over 256 initial conditions, f64. |
| `jaxmd_lj_allpairs_{energy,force}` | jax-md | `gen/gen_batch2.py`: Lennard-Jones, 4096 particles, periodic box, all pairs, f64. |
| `ml_xent_vocab32k{,_grad}` | plain JAX | `gen/gen_batch2.py`: softmax cross-entropy, batch 2048, vocab 32000, f32. |
| `ml_llama_mlp_{fwd,grad}` | plain JAX | `gen/gen_batch2.py`: RMSNorm + SwiGLU MLP, 4096 tokens, d 1024, hidden 2816, f32. |
| `ott_sinkhorn_4096{,_grad}` | ott-jax 0.6.0 | `gen/gen_batch3.py`: entropic OT between two 4096-point clouds (3D), eps 0.05, 50 Sinkhorn iterations, f32. |
| `gp_nlml_2048{,_grad}` | plain JAX | `gen/gen_batch3.py`: GP negative log marginal likelihood, RBF kernel, N=2048, Cholesky + cho_solve (pure HLO), f32. |
| `stencil_heat2d_1024` | plain JAX | `gen/gen_batch3.py`: 5-point heat stencil with `jnp.roll`, 1024^2, 50 steps in `fori_loop`. |
| `sph_density_force_4096` | plain JAX | `gen/gen_batch3.py`: SPH density + pressure force, cubic spline kernel, 4096 particles, all pairs. |
| `raytrax_w7x_trace{,_grad}` | raytrax 0.4.0 + vmecpp | `gen/gen_batch4.py raytrax` (`.venv-more`, `EQX_ON_ERROR=nan`): 140 GHz O-mode ECRH beam traced through the W7-X equilibrium (adaptive ODE on interpolated fields), optical depth and its gradient w.r.t. the launch direction, f64. |
| `brax_mjx_ant_{step,rollout16}_1024` | brax 0.14.2, MuJoCo MJX | `gen/gen_batch4.py brax`: `ant` env on the MJX backend, one step vmapped over 1024 envs, and a 16-step scan rollout, f32. |
| `jaxlie_se3_compose_log_1M`, `jaxlie_fk16_65536{,_grad}` | jaxlie | `gen/gen_batch4.py jaxlie`: SE(3) compose/inverse/log on 1M poses; 16-joint serial forward kinematics on 65536 configurations and its gradient, f32. |
| `flax_gpt4_train_grad`, `flax_resnetish_fwd` | flax 0.12.10 (linen) | `gen/gen_batch4.py flax`: 4-layer GPT (d=256, 8 heads, seq 256, batch 8, vocab 8192) loss gradient; residual CNN forward (32x64x64x3). |
| `exponax_ks2d_rollout{,_grad}` | exponax 0.2.0 (equinox 0.13.8) | `gen/gen_batch5.py exponax` (`.venv-more`): 2D Kuramoto-Sivashinsky `ex.stepper.KuramotoSivashinsky(2, 60.0, 256, 0.5)` (ETDRK, FFT-based), 256x256, f32, 100 steps in `lax.scan`; `_grad` = gradient of the MSE of the final state vs. a random target w.r.t. the initial condition. |
| `jraph_gn2_{fwd,grad}` | jraph 0.0.6.dev0 | `gen/gen_batch5.py jraph`: `jraph.GraphNetwork` x2 layers (edge MLP on [e, x_s, x_r], node MLP on [x, sum of incoming e], 2-layer ReLU MLPs, residual + layer norm, `segment_sum`), hidden 64, random graph with 50k nodes / 500k edges, f32; `_grad` = gradient of mean(h^2) w.r.t. the MLP parameters. |
| `kuramoto_rk4_{polar,cartmean}_262144_f64` | plain JAX | `gen/gen_kuramoto.py`: mean-field Kuramoto, 262144 oscillators, 200 RK4 steps in `lax.scan`, f64. Each stage reduces sin and cos of the stage phases, and the next stage depends on both sums. `polar` uses hypot/atan2 of the means; `cartmean` uses `omega + K*(S*cos - C*sin)` with `jnp.mean`, which XLA main compiles into materialized sin/cos kernels plus separate YNN reductions per stage (~4x slower than the same model with `sum * K/N`). See `repro_stage_reduce/`. |
| `kuramoto_rk8_{cartmean,cartsum}_262144_f64` | plain JAX | `repro_stage_reduce/kuramoto.py hlo cart{mean,sum}:0x262144:f64:50:rk8`: same model, DOP853's 12-stage 8th-order tableau (fixed step, no error control), 50 steps in `lax.scan` (600 RHS evaluations, vs 800 for the RK4 module), f64. `cartsum` scales the sums by K/N after the broadcast, `cartmean` uses `jnp.mean`. On XLA main neither fuses a whole step: `cartsum` gets one YNN fusion per stage (10 per step), `cartmean` materializes sin/cos per stage (24 YNN reductions per step). See `repro_stage_reduce/rk8/`. |

Venvs: `.venv-more` (jax 0.11.2: raytrax, vmecpp, brax, mujoco-mjx, jaxlie, flax, exponax, jraph), `.venv-libs` (jax 0.11.2: equinox, optax, torax, jax-fem + basix;
gmsh is stubbed because `box_mesh` does not need it), `.venv-desc`
(jax 0.9.2, DESC pins `jax<0.10`).

Dumps: `XLA_FLAGS=--xla_dump_to=<dir>`; the largest
`*.before_optimizations.txt` modules without `custom_call_target` were kept.

Runtime sweep: `./sweep_libs.sh <label> <binary> "<env>" $(cat libs_files.txt)`
(BM_HloModule only; torax_iterhybrid_step takes ~85 s to compile).
