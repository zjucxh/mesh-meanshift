import meshio
import numpy as np
import dualmesh as dm
from mesh import write_mesh
import matplotlib.pyplot as plt
from gdist import compute_gdist as geo_dist

mesh = meshio.read('./data/polyhedron.obj')
#points = mesh.points
#faces = mesh.cells[0][1]
#dual_mesh = dm.get_dual(mesh)
points = np.array(mesh.points,dtype=np.float64)
faces = np.array(mesh.cells[0][1],dtype=np.int32)

#print(faces)
#meshio.write('data/dual-bunny.obj', dual_mesh)

# geo-distance of mesh
src = np.array([0],dtype=np.int32)
dst = np.array([0,40],dtype=np.int32)
distance = geo_dist(points, faces, source_indices=src, target_indices=dst)
print('distance = {0}'.format(distance))
#write to file 

#write_mesh('data/output.obj', dual_mesh)
