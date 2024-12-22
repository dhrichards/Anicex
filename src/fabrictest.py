#%%
import os
os.environ["PETSC_DIR"] = "/usr/lib/petscdir/petsc-complex"
os.environ["SLEPC_DIR"] = "/usr/lib/petscdir/slepc-complex"
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import stokes
from Rheology import flowlaws
from params import params
from specfabufl import fabric
import utilities
import numpy as np  
import boundaryconditions as bc
import orientationtensors as ot


nx = 8; ny = 8; nz = 8

msh = mesh.create_box(MPI.COMM_WORLD,
                            [np.array([0, 0, 0]), np.array([1, 1, 1])],
                            [nx,ny,nz])



def u_soln(x):
    vals = np.zeros((msh.geometry.dim, x.shape[1]))
    vals[0] = x[1]
    return vals


par = params(dt = 1.0, surface_slope=0.5)
rheo = flowlaws.Rheology()
fab = fabric(4,0.2,1.0,0.0,0.0)

bc_u = lambda V: []
bc_f = lambda V: []

model = stokes.solver(msh, [bc_u,bc_f], par, rheo, fab)

model.u.interpolate(u_soln)


for i in range(50):
    model.solve_fabric_complex()
    a2 = ot.a2calc(model.f)
    utilities.write_xdmf("test" + str(i) + ".xdmf", 
                         msh, [model.u, a2, model.fabric.SR(model.u)],
                           ["u", "a2", "SR"], t=i)




# %%
