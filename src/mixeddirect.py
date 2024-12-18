#%%
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import numpy as np

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, la
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner, sym

opts = PETSc.Options()
opts["mat_superlu_dist_iterrefine"] = True


# We create a {py:class}`Mesh <dolfinx.mesh.Mesh>`, define functions for
# locating geometrically subsets of the boundary, and define a function
# for the  velocity on the lid:

# +
# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [32, 32], CellType.triangle
)


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


def ε(u):
    return sym(grad(u))

def η(u):
    n=2.5
    return (inner(ε(u), ε(u))/2 + 1.e-8)**((1-n)/(2*n))


# -

# Two {py:class}`function spaces <dolfinx.fem.FunctionSpace>` are
# defined using different finite elements. `P2` corresponds to a
# continuous piecewise quadratic basis (vector) and `P1` to a continuous
# piecewise linear basis (scalar).


P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
V, Q = functionspace(msh, P2), functionspace(msh, P1)

# Boundary conditions for the velocity field are defined:

# Create the Taylot-Hood function space
TH = mixed_element([P2, P1])
W = functionspace(msh, TH)

# No slip boundary condition
W0 = W.sub(0)
Q, _ = W0.collapse()
noslip = Function(Q)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
dofs = locate_dofs_topological((W0, Q), 1, facets)
bc0 = dirichletbc(noslip, dofs, W0)

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(Q)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
dofs = locate_dofs_topological((W0, Q), 1, facets)
bc1 = dirichletbc(lid_velocity, dofs, W0)

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]

# Define variational problem
# (u, p) = ufl.TrialFunctions(W)
xh = Function(W)
(u, p) = ufl.split(xh)
(v, q) = ufl.TestFunctions(W)
f = Function(Q)


F = (η(u)*inner(ε(u),ε(v)) + inner(p, div(v)) + inner(div(u), q)) * dx - inner(f, v) * dx



problem = fem.petsc.NonlinearProblem(F, xh, bcs=bcs)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
# solver.convergence_criterion = "incremental"
# solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2

# # We can customize the linear solver used inside the NewtonSolver by
# # modifying the PETSc options
# ksp = solver.krylov_solver
# opts = PETSc.Options()  # type: ignore
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "preonly"
# opts[f"{option_prefix}pc_type"] = "lu"
# sys = PETSc.Sys()  # type: ignore
# # For factorisation prefer superlu_dist, then MUMPS, then default
# if sys.hasExternalPackage("superlu_dist"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
# elif sys.hasExternalPackage("mumps"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# ksp.setFromOptions()

r = solver.solve(xh)


(u, p) = xh.split()
with XDMFFile(MPI.COMM_WORLD, "out_stokes/nonlinearvelocity.xdmf", "w") as ufile_xdmf:
        u.x.scatter_forward()
        P1 = element(
            "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type
        )
        u1 = Function(functionspace(msh, P1))
        u1.interpolate(u)
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u1)