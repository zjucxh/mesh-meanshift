import numpy as np
import open3d as o3d
from gdist import compute_gdist as goe_dist
from scipy import sparse
from mesh import farthest_point_sampling as fps
from mesh import sparse_adjacency_matrix
from mesh import range_query
from mesh import normalize
from numpy import pi
'''
def kernel(vertices, triangles, vertex_normals, c, neighbours, radius):
    #neighbours = range_query(vertices, triangles, c, radius)
    x = vertices[c]
    npos = vertices[neighbours] # neighbouring vertex positions
    nnormals = vertex_normals[neighbours]# neighbouring vertex normals
    n = len(neighbours)
    features = np.append(npos, nnormals, axis=1)
    nx = vertex_normals[c]
    x = np.append(x, nx, axis=1)[0]
    assert(len(features) > 0)
    return 1.0/pi * np.exp(-0.5 * np.linalg.norm(features-x/radius, axis=1)**2)
'''
# Kernels
def kernel(features, center, radius):
    return 1.0/pi * np.exp(-0.5 * np.linalg.norm(features-center/radius, axis=1)**2)
# Given center and grad return closest point indices
def closest_vertex_indices(x, grad, neighbours, features):
    minimum_distance = np.Inf
    idx = -1
    new_x = x + grad
    for i in range(len(features)):
        dist = np.linalg.norm(new_x - features[i])
        if minimum_distance >= dist:
            minimum_distance = dist
            idx = i
    return neighbours[idx]

# Compute gradient
def gradient(vertices, triangles, vertex_normals, c, radius):
    neighbours = range_query(vertices, triangles, c, radius)
    print('neighbours = {0}'.format(neighbours))
    x = vertices[c]
    npos = vertices[neighbours] # neighbouring vertex positions
    nnormals = vertex_normals[neighbours]
    n = len(neighbours)
    features = np.append(npos, nnormals, axis=1)
    nx = vertex_normals[c]
    x = np.append(x, nx, axis=1)[0]
    assert(len(features) > 0)
    kernel_weight = kernel(features, x, radius)  
    sum_weight = sum(kernel_weight)
    weighted_features = np.zeros_like(features, dtype=np.float32)
    for i in range(len(kernel_weight)):
        weighted_features[i] = features[i] * kernel_weight[i]
    #print(len(weighted_features))
    sum_weighted_features = np.sum(weighted_features, axis=0)
    #print(len(sum_weighted_features))
    #print(sum_weight)
    grad = sum_weight * (sum_weighted_features / sum_weight - x)
    grad = 2.0/n/radius**(2+len(grad)) * grad 
    #grad = 2.0/n/radius**(2) * grad * 100
    ind = closest_vertex_indices(x, grad, neighbours, features) 
    #print(np.linalg.norm(grad))
    return grad, ind 
    #print(sum_weighted_features)
# Compute gradient
#def gradient()
# given a mesh, starting index, and radius, return indices of converged vertices
def mean_shift(mesh, starting_ind, radius, max_iter = 20):
     
    # compute vertex normals
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    # get vertices and triangles as array
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles,dtype=np.int32)

    
    
    # Meanshift Loop
    con_vertices = np.array([], dtype=np.int32)
    cur_iter = 0 
    grad, ind = gradient(vertices, triangles, vertex_normals, starting_points, radius)

    while True:

        print('grad norm = {0}'.format(np.linalg.norm(grad)))
        if np.linalg.norm(grad) < 1.0e-6 or cur_iter >= max_iter:
            break
        grad, ind = gradient(vertices, triangles, vertex_normals, [ind], radius)
        vertices[ind] = grad[0:3] + vertices[ind]
        vertex_normals[ind] = grad[3:] + vertex_normals[ind]
        con_vertices = np.append(con_vertices, ind)     
        cur_iter = cur_iter+1
        print('Iteration {0}:\n grad = {1}\n next point = {2}'.format(cur_iter, grad, ind))
    return con_vertices


if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('./data/bunny.obj')
    

    #converged_vertices = mean_shift(mesh, [0], 3)
    vertices = np.asarray(mesh.vertices,dtype=np.float64)
    vertices = normalize(vertices) 
    triangles= np.asarray(mesh.triangles, dtype=np.int32)
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals,dtype=np.float64)
    path = mean_shift(mesh, [2], 0.2, max_iter=10)
    print(path)
