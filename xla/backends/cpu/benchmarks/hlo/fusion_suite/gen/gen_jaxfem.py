"""jax-fem Neo-Hookean hyperelasticity on a HEX8 box, solved with dumps."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_lapack  # noqa: F401
import types; sys.modules["gmsh"] = types.ModuleType("gmsh")  # box_mesh does not need gmsh
import jax, jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import box_mesh, Mesh, get_meshio_cell_type

class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F):
            E, nu = 10., 0.3
            mu, kappa = E / (2. * (1. + nu)), E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J ** (-2. / 3.)
            I1 = np.trace(F.T @ F)
            return (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.) ** 2.
        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad):
            return P_fn(u_grad + np.eye(self.dim))
        return first_PK_stress

n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
meshio_mesh = box_mesh(n, n, n, 1., 1., 1.)
cell_type = get_meshio_cell_type("HEX8")
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
left = lambda p: np.isclose(p[0], 0., atol=1e-5)
right = lambda p: np.isclose(p[0], 1., atol=1e-5)
zero = lambda p: 0.
disp = lambda p: 0.1
bcs = [[left] * 3 + [right] * 3, [0, 1, 2] * 2, [zero] * 3 + [disp, zero, zero]]
problem = HyperElasticity(mesh, vec=3, dim=3, ele_type="HEX8", dirichlet_bc_info=bcs)
sol = solver(problem, solver_options={"jax_solver": {}})
print("solved", n)
