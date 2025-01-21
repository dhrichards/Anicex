#%%
import numpy as np
import gmsh

# load tab spaced data

def loadhaut(filename):
    with open(filename) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append([float(x) for x in line.split()])

    return np.array(data)

data = loadhaut('arolla100.dat')

bottom = np.zeros((data.shape[0]-2,2))
bottom[:,0] = data[1:-1,0]
bottom[:,1] = data[1:-1,1]

top = np.zeros((data.shape[0]-2,2))
top[:,0] = data[1:-1,0]
top[:,1] = data[1:-1,2]

#flip top
top = top[::-1]

start = data[0,:2]
end = data[-1,:2]


gmsh.initialize()

gmsh.model.add("haut")

meshsize = 30

# add points
gmsh.model.geo.addPoint(start[0], start[1], 0, meshsize)

for i in range(bottom.shape[0]):
    gmsh.model.geo.addPoint(bottom[i,0], bottom[i,1], 0, meshsize)

gmsh.model.geo.addPoint(end[0], end[1], 0, meshsize)
    
for i in range(bottom.shape[0]):
    gmsh.model.geo.addPoint(top[i,0], top[i,1], 0, meshsize)



# add lines
gmsh.model.geo.addLine(1, 2, 1)
for i in range(data.shape[0]-3):
    gmsh.model.geo.addLine(i+2, i+3, i+2)

gmsh.model.geo.addLine(data.shape[0]-1, data.shape[0], data.shape[0]-1)

# add curve loop
gmsh.model.geo.addCurveLoop([i+1 for i in range(data.shape[0]-1)])

# add top line
for i in range(data.shape[0]-2):
    gmsh.model.geo.addLine(data.shape[0]+i, data.shape[0]+i+1, data.shape[0]+i)

gmsh.model.geo.addLine(data.shape[0]*2-2, 1, data.shape[0]*2-2)

# add curve loop
gmsh.model.geo.addCurveLoop([i+1 for i in range(data.shape[0]-1, 2*data.shape[0]-2)])

# add surface
gmsh.model.geo.addPlaneSurface([1,2])

## synchronize
gmsh.model.geo.synchronize()

## add physical groups for top and bottom lines
gmsh.model.addPhysicalGroup(1, [1], 1)
gmsh.model.setPhysicalName(1, 1, "bottom")

gmsh.model.addPhysicalGroup(1, [2], 2)
gmsh.model.setPhysicalName(1, 2, "top")

gmsh.model.geo.synchronize()

# mesh
gmsh.model.mesh.generate(2)


# save geo file
gmsh.write("haut.geo_unrolled")

# save msh file
gmsh.write("haut.msh")

#finalize
gmsh.finalize()
# %%
