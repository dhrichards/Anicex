#%%
import os
os.environ["PETSC_DIR"] = "/usr/lib/petscdir/petsc-complex"
os.environ["SLEPC_DIR"] = "/usr/lib/petscdir/slepc-complex"

import specfabufl
import ufl
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import basix.ufl as bufl

msh = mesh.create_unit_cube(MPI.COMM_WORLD, 2,2,2)

def u_soln(x):
    vals = np.zeros((msh.geometry.dim, x.shape[1]))
    vals[0] = x[2]
    return vals

P2_el = bufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
V = fem.functionspace(msh, P2_el)

Q = fem.functionspace(msh, ("Lagrange", 1))

M_el = bufl.element("Lagrange", msh.basix_cell(), 1, shape=(15,15))
M_space = fem.functionspace(msh, M_el)

u = fem.Function(V)

u.interpolate(u_soln)


# gradu = np.array([[0, 1., 0], 
#                   [0 ,0, 0], 
#                   [0, 0, 0]])

# D = 0.5*(gradu + gradu.T)
# W = 0.5*(gradu - gradu.T)

# ε = ufl.as_matrix(D)
# ω = ufl.as_matrix(W)

ε = ufl.sym(ufl.grad(u))
ω = ufl.skew(ufl.grad(u))

# Mufl = specfabufl.M_LROT(4, ε, ω, 1.0)
Mufl = specfabufl.M_REG(4, ε)

m = fem.Function(Q)

Mnp = np.zeros((15,15), dtype=np.complex64)
for i in range(15):
    for j in range(15):
        expr = fem.Expression(Mufl[i,j],Q.element.interpolation_points())
        m.interpolate(expr)
        # print(m.x.array[0])
        Mnp[i,j] = m.x.array[0]


np.save("Mnp.npy", Mnp)
# np.save("gradu.npy", gradu)
    

# expr = fem.Expression(Mufl[0,0],Q.element.interpolation_points())

# m.interpolate(expr)

# # import utilities

# # utilities.write_xdmf("test.xdmf", msh, [m], ["m"], t=0)
# print(m.x.array[:])