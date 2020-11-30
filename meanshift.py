import numpy as np
import open3d as o3d
from gdist import compute_gdist as goe_dist
from scipy import sparse
from mesh import farthest_point_sampling as fps
from mesh import sparse_adjacency_matrix
from mesh import range_query

def ker_dist(x):
    return x**2
def ker_norm(x):
    return x**2


# given a mesh, starting pints, and radius, return indices of converged vertices
def mean_shift(mesh, starting_points, radius):
    #TODO normalize mesh

    # compute vertex normals
    mesh.compute_vertex_normals()
    vertex_normals = mesh.vertex_normals

    # compute adjacency matrix
    adjacency = sparse_adjacency_matrix(mesh)
    
    converged = False

    while not converged:

    con_vertices = np.array([], dtype=np.int32)
    return con_vertices


if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('./data/polyhedron.obj')
    converged_vertices = mean_shift(mesh, [0], 3)

