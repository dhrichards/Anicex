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



L = 5e3 # length of 1 wavelength
H = 1e3
repetitions = 8
Lmax = 4

nz = 10
nx = int(nz*L*repetitions/H)

def top(x):
    return H

def bottom(x):
    return 0.5*H*np.sin(x*2*np.pi/L)

def bottom_boundary(x):
    r = -x[1] + bottom(x[0])
    return np.isclose(r,0.0)

def left_boundary(x):
    return np.isclose(x[0],-L*repetitions//2)

def right_boundary(x):
    return np.isclose(x[0],L*repetitions//2)

def top_boundary(x):
    return np.isclose(x[1],H)


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([-L*repetitions//2, 0]), np.array([L*repetitions//2, 1])],
                            [nx,nz], mesh.CellType.quadrilateral)


def warp_mesh(msh,top,bot):
    x = msh.geometry.x[:,0]
    z = msh.geometry.x[:,1]
    zs = top(x)
    zb = bot(x)
    znew = zb + z*(zs-zb)
    xynewcoor = np.array([x, znew]).transpose()
    msh.geometry.x[:,:2] = xynewcoor



warp_mesh(msh,top,bottom)







par = params(dt = 5.0, surface_slope=0.5)
rheo = flowlaws.Rheology(flowlaw='Sachs')
fab = fabric(4,0.3,1.0,0.0,0.0, D = msh.geometry.dim)

bc_u = lambda V: [bc.get_zero_bc(V, bottom_boundary),
                  bc.get_zero_bc(V, left_boundary),
                  bc.get_zero_bc(V, right_boundary)]
bc_f = lambda V: []
bc_z = lambda V: []

model = stokes.solver(msh, [bc_u,bc_f,bc_z], par, rheo, fab, top_boundary)

# model.solve_stokes()

# utilities.write_xdmf("test.xdmf", msh, [model.u], "u")

#%%
import orientationtensors as ot
for i in range(200):
    model.solve_stokes()
    model.solve_fabric_complex()
    # model.move_mesh()
    a2 = ot.a2calc(model.f)
    utilities.write_xdmf("test" + str(i) + ".xdmf", 
                         msh, [model.u, a2], ["u", "a2"], t=i)






    

