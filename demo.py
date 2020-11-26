import meshio
import numpy as np
import dualmesh as dm
from mesh import write_mesh
import matplotlib.pyplot as plt
from gdist import compute_gdist as geo_dist
import gdist
mesh = meshio.read('./data/cube.obj')
#points = mesh.points
#faces = mesh.cells[0][1]
#dual_mesh = dm.get_dual(mesh)
points = np.array(mesh.points,dtype=np.float64)
faces = np.array(mesh.cells[0][1],dtype=np.int32)
print(faces)
#print(faces)
#meshio.write('data/dual-bunny.obj', dual_mesh)

# geo-distance of mesh
#dst = np.array(range(len(points)),dtype=np.int32)
#src = np.array([0],dtype=np.int32)
#dist_array = geo_dist(points, faces, source_indices=src, target_indices=dst)
#print('pair wise distance = {0}'.format(dist_array))
#src = np.array([1],dtype=np.int32)
#dist_array = geo_dist(points, faces, source_indices=src, target_indices=dst)
#print('pair wise distance = {0}'.format(dist_array))
#src = np.array([2],dtype=np.int32)
#dist_array = geo_dist(points, faces, source_indices=src, target_indices=dst)
#print('pair wise distance = {0}'.format(dist_array))
#src = np.array([3],dtype=np.int32)
#dist_array = geo_dist(points, faces, source_indices=src, target_indices=dst)
#print('pair wise distance = {0}'.format(dist_array))

src = np.array([0,1,2,3],dtype=np.int32)
dst = np.array(range(len(points)),dtype=np.int32)
dist_array = gdist.distance_matrix_of_selected_points(points, faces, src)
print('pair wise distance = {0}'.format(dist_array))
#src = np.array(range(len(points)), dtype=np.int32)
#src = np.array([2,5,10], dtype=np.int32)
#print('distance = {0}'.format(distance))
# pair wise distance
geo_dist_matrix = gdist.local_gdist_matrix(vertices=points, triangles=faces,max_distance=np.float('Inf')) 
print('pair-wise matrix = {0}\n'.format(geo_dist_matrix))
#write to file 

#write_mesh('data/output.obj', dual_mesh)
