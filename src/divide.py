#%%
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



L = 10e3 # length of 1 wavelength
H = 1e3
Lmax = 4



nz = 10
nx = int(nz*L/H)



def bottom_boundary(x):
    return np.isclose(x[1],0.0)

def top_boundary(x):
    return np.isclose(x[1],H)

def left_boundary(x):
    return np.isclose(x[0],0.0)

def right_boundary(x):
    return np.isclose(x[0],L)


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([L, H])],
                            [nx,nz], mesh.CellType.quadrilateral)




par = params(dt = 0.001, surface_slope=0.0)
rheo = flowlaws.Rheology(flowlaw='Glen')
fab = fabric(4,0.3,1.0,0.0,0.0, D = msh.geometry.dim)

bc_u = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                bc.get_zero_bc(V, bottom_boundary)]
bc_f = lambda V: []
bc_z = lambda V: [bc.get_zero_bc(V, bottom_boundary)]
model = stokes.solver(msh, [bc_u,bc_f,bc_z], par, rheo, fab, top_boundary)

model.solve_stokes()

model.move_mesh()

utilities.write_xdmf("test.xdmf", msh, [model.u, model.z], ["u", "z"], t=0)

