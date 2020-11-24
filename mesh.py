import numpy as np
import meshio
from gdist import compute_gdist as geo_dist
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# visualize FPS algorithms
def fps_vis(vertices, samples):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]
    cx = [x[i] for i in range(len(x)) if i not in samples]
    cy = [y[i] for i in range(len(y)) if i not in samples]
    cz = [z[i] for i in range(len(z)) if i not in samples]
 
    ax.scatter(cx,cy,cz, c='r',marker='o')
    sx = [x[i] for i in samples] 
    sy = [y[i] for i in samples] 
    sz = [z[i] for i in samples] 
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
    # for debug 
    #idx = 0
    idx = np.int32(idx)
    samples = np.append(samples, idx)
    for e in samples:
        mask[e] = False
    # TODO compute farthest point and append to samples
    #remaining_index = np.array([e for e in vertex_indices if e not in samples], dtype=np.int32)
    #print(remaining_index)
    #dist_array = geo_dist(vertices, triangles, source_indices=samples)
    # calculate farthes point to samples
    dist_array = geo_dist(vertices, triangles, source_indices=samples, target_indices=vertex_indices)
    for i in range(1, len(dist_array)):
        if mask[i]==False:
            dist_array[i] = np.float64('-Inf')    
    max_idx = np.argmax(dist_array)
    samples = np.append(samples, np.int32(max_idx))
    mask[max_idx] = False
    #print(len(vertices))
    # TODO compute remianing points set distance for each sample,select the nearest 
    for iter in range(num_sample-2):

        dist_array = np.array([], dtype=np.float64)
        for i in range(len(vertex_indices)):
            #print('samples = {0}'.format(samples))
            dist = geo_dist(vertices, triangles, source_indices=vertex_indices[i:i+1], target_indices=samples)
            #print('dist = {0}'.format(dist))
            dist_array = np.append(dist_array, dist)
        dist_array = np.reshape(dist_array, (-1, len(samples)))
        #print('dist_array = {0}'.format(dist_array))
        max_min = max_min_args(dist_array, samples) 
        samples = np.append(samples, np.int32(max_min))
        mask[max_min] = False
    print('samples = {0}'.format(samples))
    print('len samples = {0}'.format(len(samples)))
    fps_vis(vertices, samples)
        #print(max_min)


    # TODO select farthes in remaining points

if __name__=='__main__':

    mesh = meshio.read('./data/half_polyhedron.obj')
    vertices = np.array(mesh.points, dtype=np.float64)
    triangles = np.array(mesh.cells[0][1], dtype=np.int32)
    farthest_point_sampling(vertices, triangles, 15)
    #fps_vis(vertices, samples)
