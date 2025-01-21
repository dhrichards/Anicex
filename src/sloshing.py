#%%
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import ufl.mathfunctions
import stokes
from Rheology import flowlaws
from params import params
from specfabufl import fabric
import utilities
import numpy as np  
import boundaryconditions as bc



Lx = 100e3 # length of 1 wavelength
Lz = 1e3
z0 = 0.1e3

Lmax = 4



nz = 10
nx = 50

def top(x):
    return Lz + z0*np.cos(np.pi*x/Lx)

def bottom(x):
    return 0.0





def bottom_boundary(x):
    return np.isclose(x[1],0.0)

def left_boundary(x):
    return np.isclose(x[0],0)

def right_boundary(x):
    return np.isclose(x[0],Lx)

def top_boundary(x):
    r = -x[1] + top(x[0])
    return np.isclose(r,0.0)


msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [np.array([0, 0]), np.array([Lx, 1])],
                            [nx,nz], mesh.CellType.triangle)

def warp_mesh(msh,top,bot):
    x = msh.geometry.x[:,0]
    z = msh.geometry.x[:,1]
    zs = top(x)
    zb = bot(x)
    znew = zb + z*(zs-zb)
    xynewcoor = np.array([x, znew]).transpose()
    msh.geometry.x[:,:2] = xynewcoor



warp_mesh(msh,top,bottom)



par = params(dt = 100)
rheo = flowlaws.Rheology(flowlaw='Glen')
fab = fabric(4,0.3,1.0,0.0,0.0, D = msh.geometry.dim)


facets = mesh.locate_entities_boundary(msh, 1, left_boundary)
bc_u = lambda V: [bc.get_zero_bc(V, bottom_boundary),
                    bc.get_zero_bc(V.sub(0), left_boundary),
                    bc.get_zero_bc(V.sub(0), right_boundary)]
bc_f = lambda V: []
bc_z = lambda V: [bc.get_zero_bc(V, bottom_boundary)]



model = stokes.solver(msh, [bc_u,bc_f,bc_z], par, rheo, fab, top_boundary)

model.setup_stokes(FSSA=True)



# model.solve_stokes()
# model.move_mesh()

utilities.write_file("outputs/slosh.xdmf", msh, [model.u,model.z], ["u","z"])


#%%
import orientationtensors as ot
for i in range(100):
    if MPI.COMM_WORLD.rank == 0:
        print(i)
    model.solve_stokes()

    utilities.write_file("outputs/slosh" + str(i) + ".xdmf", 
                         msh, [model.u, model.z], ["u", "z"], t=i*par.dt)
    
    # model.solve_fabric_complex()
    model.move_mesh()
    # a2 = ot.a2calc(model.f)

    # model.msh.geometry.x[:,model.msh.geometry.dim-1] = model.z_original.x.array.real
    # model.msh.geometry.x[:,model.msh.geometry.dim-1] = model.z.x.array.real




    
#%%

# msh = model.msh

# Q = fem.functionspace(msh, ("Lagrange", 1, model.u.ufl_shape))
# u = fem.Function(Q)
# u.interpolate(fem.Expression(model.u,Q.element.interpolation_points()))
# x = msh.geometry.x
# vals = np.zeros((x.shape[0], 3))
# vals[:,:len(u)] = u.x.array.reshape((x.shape[0], len(u)))

# from matplotlib import pyplot as plt

# # unique x vals
# xvals = np.unique(x[:,0])

# # get largest y val for each unique x value
# yvals = np.zeros_like(xvals)
# uvals = np.zeros_like(xvals)

# for i, xval in enumerate(xvals):
#     yvals[i] = np.max(x[x[:,0] == xval,1])

# # get idx of largest y val for each unique x value
# for i, xval in enumerate(xvals):
#     idx = np.argmax(x[x[:,0] == xval,1])
#     uvals[i] = vals[x[:,0] == xval][idx,0]




# plt.plot(xvals, yvals)