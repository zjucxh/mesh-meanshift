import numpy as np
from sklearn.cluster import SpectralClustering, MeanShift, estimate_bandwidth
from sklearn import metrics
from gdist import local_gdist_matrix as gdist_matrix
from gdist import compute_gdist as geo_distance
import open3d as o3d
from scipy import sparse

# visualize
def render_mesh(mesh):
    #mesh.paint_uniform_color([1,0.4,1])
    #color = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],dtype=np.float)
    o3d.visualization.draw_geometries([mesh]) 

# compute distance from point to multiple point
def distance(vertices, vertex_normals, src_ind=0, dst_ind = np.array([0], dtype=np.int32)):
    src = vertices[src_ind]
    dst = vertices[dst_ind]
    abs_dist = np.abs(dst - src)
    #norm_dist = np.abs(vertex_normals[dst_ind],vertex_normals[src_ind])

    return np.sum(abs_dist,axis=1)
# fetch data
mesh = o3d.io.read_triangle_mesh('./data/garment.off')
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices,dtype=np.float64)
triangles = np.asarray(mesh.triangles,dtype=np.int32)
vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)


#Compute Adjacency Matrix
mesh.compute_adjacency_list()
if mesh.has_adjacency_list():
    adjacency_list = mesh.adjacency_list
adjacency_array = np.zeros(shape=(len(vertices), len(vertices)))

for i in range(len(adjacency_list)):
    e = np.array(list(adjacency_list[i]), dtype=np.int32)
    
    dist = distance(vertices,vertex_normals, i, e) 
    adjacency_array[i,e] = dist
print('adjacency matrix calculated')

#Spectral Method
n_clusters = 29
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_jobs=4)
sc.fit(adjacency_array)
labels = sc.labels_

'''
# Meanshift method
features = np.append(vertices, vertex_normals, axis=1)
#bandwidth = 0.35
bandwidth = estimate_bandwidth(features, quantile=0.1)
print('band width = {0}'.format(bandwidth))
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(features)
labels = ms.labels_
'''
n_labels = len(np.unique(labels))
print('number of clusters {0}'.format(n_labels))

# Visualize
r = np.linspace(0.0, 1.0, n_labels)
g = np.abs(np.linspace(1.0, 0.0, n_labels))
b = np.abs(np.sin(np.linspace(0, 2.0 * np.pi, n_labels)))
r = np.reshape(r, (-1, 1))
g = np.reshape(g, (-1, 1))
b = np.reshape(b, (-1, 1))
color_map = np.append(r, g, axis = 1)
color_map = np.append(color_map, b, axis=1)
#print('color map: {0}'.format(color_map))
color = np.zeros((len(labels), 3), dtype=np.float64)
ii = 0
for i in labels:
    color[ii] = color_map[i]
    ii = ii + 1
mesh.vertex_colors = o3d.utility.Vector3dVector(color)
print('has vertex colors: {0}'.format(mesh.has_vertex_colors()))
render_mesh(mesh) 
