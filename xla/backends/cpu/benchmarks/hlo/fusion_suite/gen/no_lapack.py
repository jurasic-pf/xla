"""Import before tracing: lower LU, triangular solve and Cholesky with JAX's
pure-HLO rules on CPU, so exported modules have no LAPACK custom calls."""
from jax._src.interpreters import mlir
from jax._src.lax import linalg as L

mlir.register_lowering(L.lu_p, mlir.lower_fun(L._lu_python, multiple_results=True), platform="cpu")
mlir.register_lowering(L.triangular_solve_p, L._triangular_solve_lowering, platform="cpu")
if hasattr(L, "_cholesky_lowering"):
    mlir.register_lowering(L.cholesky_p, L._cholesky_lowering, platform="cpu")
