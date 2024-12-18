import numpy as np
import stokessolver
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from mpi4py import MPI



# ## Define the mesh and boundary conditions


def createStructuredMesh(xvec, bottomsurf, topsurf, Nx, Nz):
    hl = Nx-1 # Number of horizontal layers
    vl = Nz-1 # Number of vertical layers
    # generate mesh on unitsquare
    mesh = create_rectangle(
        MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [Nx, Nz], CellType.triangle
    )
    
    # Extract x and y coordinates from mesh
    x = mesh.geometry.x[:, 0]
    y = mesh.geometry.x[:, 1]
    x0 = min(xvec)
    x1 = max(xvec)
    
    # Map coordinates on unit square to the computational domain
    xnew = x0 + x*(x1-x0)
    zs = np.interp(xnew, xvec, topsurf)
    zb = np.interp(xnew, xvec, bottomsurf)
    ynew = zb + y*(zs-zb)
    
    xynewcoor = np.array([xnew, ynew]).transpose()
    mesh.coordinates()[:] = xynewcoor
    
    return mesh