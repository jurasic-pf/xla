"""Small DESC equilibrium solve with XLA dumps (jax 0.9 venv)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import no_lapack  # noqa: F401
except Exception as e:
    print("no_lapack not applied:", e)
import desc.examples
from desc.equilibrium import Equilibrium
eq = desc.examples.get("HELIOTRON")
eq.change_resolution(L=8, M=8, N=3, L_grid=12, M_grid=12, N_grid=6)
eq.solve(maxiter=2, verbose=0)
print("solved")
