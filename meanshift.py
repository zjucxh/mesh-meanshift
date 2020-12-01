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

# Compute gradient
def gradient(vertices, triangles, vertex_normals, c, radius):
    neighbours = range_query(vertices, triangles, c, radius)
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
    print(grad) 
    print(np.linalg.norm(grad))
    return grad 
        
    #print(sum_weighted_features)

# given a mesh, starting pints, and radius, return indices of converged vertices
def mean_shift(mesh, starting_points, radius):
    #TODO normalize mesh
     
    # compute vertex normals
    mesh.compute_vertex_normals()
    vertex_normals = mesh.vertex_normals
    # get vertices and triangles as array
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles,dtype=np.int32)
    # Farthest Points Sampling 
    samples = fps(vertices, triangles)

    # Meanshift Loop
    converged = False
    '''
    while True:
        grad = gradient(vertices, triangles, vertex_normals, [0], 0.8)
        vec = vertices[0]
        vec = vec - 
    '''    
    con_vertices = np.array([], dtype=np.int32)
    return con_vertices


if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('./data/bunny.obj')

    #converged_vertices = mean_shift(mesh, [0], 3)
    vertices = np.asarray(mesh.vertices,dtype=np.float64)
    #vertices = normalize(vertices) 
    triangles= np.asarray(mesh.triangles, dtype=np.int32)
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals,dtype=np.float64)
    gradient(vertices, triangles, vertex_normals, [0], 1.2)
