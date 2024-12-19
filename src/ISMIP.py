#%%
import numpy as np
from dolfinx import mesh, fem, plot
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import stokes
from Rheology import flowlaws
from params import params
from specfabufl import fabric



L = 5e3 # length of 1 wavelength
H = 1e3
repetitions = 4
Lmax = 4

nz = 10
nx = int(nz*L*repetitions/H)

def top(x):
    return H

def bottom(x):
    return 0.5*H*ufl.sin(x[0]*2*ufl.pi/L)


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([L*repetitions, 1])],
                            [nx,nz], mesh.CellType.quadrilateral)


def warp_mesh(msh,top,bot):
    x = msh.geometry.x[:,0]
    z = msh.geometry.x[:,1]
    zs = top(x)
    zb = bot(x)
    znew = zb + z*(zs-zb)
    xynewcoor = np.array([x, znew]).transpose()
    msh.geometry.x[:] = xynewcoor


warp_mesh(msh,top,bottom)



par = params(dt = 1.0)
rheo = flowlaws.Rheology()
fab = fabric(4,0.3,1.0,0.0,0.0)

bc_u = lambda V: [fem.DirichletBC(V, ufl.as_vector([0,0]), lambda x: np.isclose(x[0],0.0))]
bc_f = lambda V: []

model = stokes.solver(msh, [bc_u,bc_f], par, rheo, fab)

model.solve_stokes()

#%%






    

