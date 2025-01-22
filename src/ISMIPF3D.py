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



L = 400e3 # length of 1 wavelength
Ly = 100e3
wavelength = 100e3
H = 1e3

nz = 8
ny = 20
nx = 120

a0 = 100
σ = 10e3

A = 2.140373e-1
n = 1
surface_slope = 3.0
Uin = 100.0
D = 3

def top(x):
    return 0.0

def bottom(x, y=0.0):
    return -H +  a0*np.exp(-(x**2 + y**2)/(σ**2))\
        + a0*np.exp(-((x-wavelength)**2 + y**2)/(σ**2)) \
        + a0*np.exp(-((x+wavelength)**2 + y**2)/(σ**2)) 


def bottom_boundary(x):
    r = -x[-1] + bottom(x[0],x[1])
    return np.isclose(r,0.0)

def left_boundary(x):
    return np.isclose(x[0],-L//2)

def right_boundary(x):
    return np.isclose(x[0],L//2)

def front_boundary(x):
    return np.isclose(x[1],-Ly//2)

def back_boundary(x):
    return np.isclose(x[1],Ly//2)

def top_boundary(x):
    return np.isclose(x[-1],0.0)


msh = mesh.create_box(MPI.COMM_WORLD,
                    [np.array([-1, -1, 0]), np.array([1, 1, 1])],
                    [nx, ny, nz], mesh.CellType.tetrahedron)


msh.geometry.x[:,0] = msh.geometry.x[:,0]*L/2
msh.geometry.x[:,1] = msh.geometry.x[:,1]*Ly/2




def warp_mesh(msh,top,bot):
    x = msh.geometry.x[:,0]
    y = msh.geometry.x[:,1]
    z = msh.geometry.x[:,2]
    zs = top(x)
    zb = bot(x,y)
    znew = zb + z*(zs-zb)
    xynewcoor = np.array([x, y, znew]).transpose()
    msh.geometry.x[:,:3] = xynewcoor





warp_mesh(msh,top,bottom)



def u_bc(x):
    return np.stack([-Uin*(x[2,:]/H)**2 + Uin, np.zeros(x.shape[1]), np.zeros(x.shape[1])])


par = params(dt = 0.4, surface_slope=surface_slope)
rheo = flowlaws.Rheology(flowlaw='Glen', B=A, n=n)
fab = fabric(4,0.3,1.0,0.0,0.0, D = msh.geometry.dim)


facets = mesh.locate_entities_boundary(msh, 1, left_boundary)
bc_u = lambda V: [bc.get_zero_bc(V, bottom_boundary),
                  bc.get_bc_func(V, right_boundary, u_bc),
                  bc.get_bc_func(V, left_boundary, u_bc),
                  bc.get_zero_bc(V.sub(1), front_boundary),
                  bc.get_zero_bc(V.sub(1), back_boundary)]

bc_f = lambda V: []
bc_z = lambda V: [bc.get_bc_func(V, bottom_boundary, lambda x: bottom(x[0],x[1]))]



model = stokes.solver(msh, [bc_u,bc_f,bc_z], par, rheo, fab, top_boundary)


model.setup_stokes(FSSA=True)
# model.solve_stokes(FSSA=True)
# model.move_mesh()

# utilities.write_vtk("testF.pvd", msh, [model.u,model.z], ["u","z"])


#%%
for i in range(500):
    if MPI.COMM_WORLD.rank == 0:
        print(i)
    model.solve_stokes()

    model.msh.geometry.x[:,model.msh.geometry.dim-1] = model.z_original.x.array.real
    utilities.write_file("outputs/expF3D" + str(i) + ".xdmf", 
                         msh, [model.u, model.z], ["u", "z"], t=i)
    model.msh.geometry.x[:,model.msh.geometry.dim-1] = model.z.x.array.real

    model.move_mesh()
    # a2 = ot.a2calc(model.f)

  




    
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
# plt.figure()
# plt.plot(xvals, uvals)
    

# #%%

# # plot over y at fixed x

# xslice = xvals[50]

# y = x[x[:,0] == xslice,1]
# u = vals[x[:,0] == xslice,0]

# u_analytic = -Uin*(y/H)**2 + Uin
# plt.plot(y,u)
# plt.plot(y,u_analytic)
