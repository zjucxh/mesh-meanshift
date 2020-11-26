import numpy as np
import meshio
from gdist import compute_gdist as geo_dist
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d



#o3d.geometry.TriangleMesh.compute_vertex_normals
# write mesh to obj
def write_mesh(filename, mesh):
    vertices = mesh.points
    faces = mesh.cells[0][1]
    f = open(filename, mode='w')
    f.write('# OBJ file format with ext .obj\n')
    f.write('# vertex count = {0}\n'.format(len(vertices)))
    f.write('# face count = {0}\n'.format(len(faces)))
    for v in vertices:
        f.write('v {0} {1} {2}\n'.format(v[0], v[1], v[2]))
    for e in faces:
        f.write('f')
        for i in range(len(e)):
            f.write(' {0}'.format(e[i]+1))
        f.write('\n')

# normalize mesh
def normalize(mesh):
    vertices = mesh.points
    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]
    max_x = np.max(x)
    max_y = np.max(y)
    max_z = np.max(z)
    min_x = np.min(x)
    min_y = np.min(y)
    min_z = np.min(z)
    x = 2.0*(x-min_x)/(max_x-min_x) - 1.0
    y = 2.0*(y-min_y)/(max_y-min_y) - 1.0
    z = 2.0*(z-min_z)/(max_z-min_z) - 1.0
    vertices[:,0]=x
    vertices[:,1]=y
    vertices[:,2]=z
    mesh.points=vertices
    return mesh


# visualize FPS algorithms
def fps_vis(vertices, samples):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]
    filter = np.array([True for e in x])
    filter[samples]=False
    cx = x[filter]
    cy = y[filter]
    cz = z[filter]
    #plt.scatter(cx,cy,cz, c='r',marker='o') 
    ax.scatter(cx,cy,cz, c='r',marker='o')
    sx = x[samples]
    sy = y[samples]
    sz = z[samples]
        
    #fig.add_subplot(122,projection='3d')
    ax.scatter(sx,sy,sz, c='g',marker='o')
    ax.set_xlabel('X label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label')
    
    plt.show()
# farthest point sampling algorithm
def max_min_args(array, samples):
    for i in range(len(samples)):
        idx = samples[i]
        array[idx][i] = np.float64('Inf') 
        #print(array)
    #for i in 
    min_arr = np.array([], dtype=np.float64)
    min_idx = np.array([], dtype=np.int32)
    for e in array:
        idx = np.argmin(e)
        min_arr = np.append(min_arr, e[idx])
        min_idx = np.append(min_idx, idx)
    for i in samples:
        min_arr[i] = np.float64('-Inf')

    #print('min_arr = {0}'.format(min_arr))
    #print('min_idx = {0}'.format(min_idx))
    #print('max min = {0}'.format(np.argmax(min_arr)))
    return np.argmax(min_arr)

def farthest_point_sampling(vertices, triangles, num_sample=15):
    vertex_indices = range(len(vertices))
    vertex_indices = np.array(vertex_indices, dtype=np.int32)
    #vertex_indices = np.random.choice(vertex_indices, 300, replace=False)
    #print('max = {0}, min = {1}'.format(np.max(vertex_indices), np.min(vertex_indices)))
    mask = [True for e in vertex_indices]
    samples = np.array([], dtype=np.int32)
    idx = np.random.randint(0, len(vertex_indices))
    idx = np.int32(idx)
    samples = np.append(samples, idx)
    for e in samples:
        mask[e] = False
    # calculate farthes point to samples
    dist_array = geo_dist(vertices, triangles, source_indices=samples, target_indices=vertex_indices)
    for i in range(1, len(dist_array)):
        if mask[i]==False:
            dist_array[i] = np.float64('-Inf')    
    max_idx = np.argmax(dist_array)
    samples = np.append(samples, np.int32(max_idx))
    mask[max_idx] = False
    for iter in range(num_sample-2):
        dist_array = np.array([], dtype=np.float64)
        for i in range(len(vertex_indices)):
            dist = geo_dist(vertices, triangles, source_indices=vertex_indices[i:i+1], target_indices=samples)
            dist_array = np.append(dist_array, dist)
        dist_array = np.reshape(dist_array, (-1, len(samples)))
        max_min = max_min_args(dist_array, samples) 
        samples = np.append(samples, np.int32(max_min))
        mask[max_min] = False
    return samples
# TODO given array of vertices and adjacency matrix, return neighbouring vertices
#def vertex_neighbors(vertices, adjacent_matrix, depths = 3):
#    return vertices

# TODO given a triangle mesh, return sparse adjacency list
# TODO eliminate waring 
def sparse_adjacency_matrix(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float)
    triangles = np.asarray(mesh.triangles, dtype=np.int)
    matrix = []
    for i in range(len(vertices)):
        matrix.append(np.array([], dtype=np.int32))
    for e in triangles:
        a, b, c = e[0], e[1], e[2]
        matrix[a] = np.append(matrix[a], [b,c])
        matrix[b] = np.append(matrix[b], [a,c])
        matrix[c] = np.append(matrix[c], [a,b])
    for e in range(len(matrix)):
        matrix[e] = np.unique(matrix[e])
    return matrix

if __name__=='__main__':

   mesh = o3d.io.read_triangle_mesh('./data/half_polyhedron.obj') 
   mesh.compute_triangle_normals()
   mesh.compute_vertex_normals()

   vertices = np.asarray(mesh.vertices,dtype=np.float64)
   triangles = np.asarray(mesh.triangles,dtype=np.int32)
   # if not mesh.has_vertex_normals():
   vertex_normals = mesh.vertex_normals
   # if not mesh.has_triangle_normals():
   triangle_normals = mesh.triangle_normals
   #samples = farthest_point_sampling(vertices, triangles)
   #fps_vis(vertices, samples)
   #o3d.visualization.draw_geometries([mesh])
   sparse_matrix = sparse_adjacency_matrix(mesh)
   print(np.asarray(sparse_matrix))
   
