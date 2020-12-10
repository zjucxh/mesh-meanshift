import numpy as np
import open3d as o3d

def parse_obje(obj_file, scale_by):
    V = np.array
    vs = []
    faces = []
    edges = []

    def add_to_edges():
        if edge_c >= len(edges):
            for _ in range(len(edges), edge_c + 1):
                edges.append([])
        edges[edge_c].append(edge_v)

    def fix_vertices():
        nonlocal vs, scale_by
        vs = V(vs)
        z = vs[:, 2].copy()
        vs[:, 2] = vs[:, 1]
        vs[:, 1] = z
        max_range = 0
        for i in range(3):
            min_value = np.min(vs[:, i])
            max_value = np.max(vs[:, i])
            max_range = max(max_range, max_value - min_value)
            vs[:, i] -= min_value
        if not scale_by:
            scale_by = max_range
        vs /= scale_by

    with open(obj_file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:]])
            elif splitted_line[0] == 'f':
                faces.append([int(c) - 1 for c in splitted_line[1:]])
            elif splitted_line[0] == 'e':
                if len(splitted_line) >= 4:
                    edge_v = [int(c) - 1 for c in splitted_line[1:-1]]
                    edge_c = int(splitted_line[-1])
                    add_to_edges()

    vs = V(vs)
    fix_vertices()
    faces = V(faces, dtype=int)
    edges = [V(c, dtype=int) for c in edges]
    return (vs, faces, edges), scale_by
def selected_part(mesh, label):
    vertices = mesh[0]
    faces = mesh[1]
    edges = mesh[2]
    selected_edges = edges[label]
    selected_vertex_indices = np.unique(np.reshape(selected_edges, (-1,)))
    selected_vertex_indices = np.array([i for i in range(len(vertices)) if i not in selected_vertex_indices],dtype=np.int32) 
    return selected_vertex_indices

if __name__=='__main__':
    mesh, scale = parse_obje('data/shrec__1_0.obj',0)
    edges = mesh[2]
    vertices = mesh[0]
    faces = mesh[1]
    part_id = 4
    part = edges[part_id]
    part = np.unique(np.reshape(part, (-1,)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices[part])
    selected_faces = selected_part(mesh, part_id)
    #selected_faces = np.array([not e for e in selected_faces],dtype=np.bool)   
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_vertices_by_index(selected_faces)
    mesh.remove_unreferenced_vertices()
    #mesh.compute_vertex_normals()
    #mesh.compute_triangle_normals()
    o3d.visualization.draw_geometries([mesh])
    # write mesh
    o3d.io.write_triangle_mesh('data/body_part.obj',mesh)
    print('completed.')
