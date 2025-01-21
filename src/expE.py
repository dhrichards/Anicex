#%%
import numpy as np
from dolfinx import mesh, fem, plot, default_scalar_type, io
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
import meshio


L = 10e3 # length of 1 wavelength
H = 1e3
Lmax = 4


## load
# msh,ct,ft = io.gmshio.read_from_msh("haut.msh", MPI.COMM_WORLD, gdim=2)

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh


msh = meshio.read("haut.msh")

# Create and save one file for the mesh, and one file for the facets
triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
line_mesh = create_mesh(msh, "line", prune_z=True)
meshio.write("mesh.xdmf", triangle_mesh)
meshio.write("mt.xdmf", line_mesh)

#%%

par = params(dt = 0.1, surface_slope=0.0)
rheo = flowlaws.Rheology(flowlaw='Glen')
fab = fabric(4,0.3,1.0,0.0,0.0, D = msh.geometry.dim)

bc_u = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary),
                bc.get_zero_bc(V, bottom_boundary)]
bc_f = lambda V: []
bc_z = lambda V: [bc.get_zero_bc(V, bottom_boundary)]
model = stokes.solver(msh, [bc_u,bc_f,bc_z], par, rheo, fab, top_boundary)


for i in range(50):

    model.solve_stokes()
    model.solve_fabric_complex()
    model.move_mesh()

    utilities.write_xdmf("test" + str(i) + ".xdmf", msh,
     [model.u, model.z, ot.a2calc(model.f)], ["u", "z", "a2"], t=i)

