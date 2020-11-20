import meshio
import numpy as np

class Mesh:
    vertices = []
    edges = np.empty((0,2), dtype=int)
    faces = []
    adjacencyMatrix = []
    
    def __init__(self, vertices, faces):
        self.vertices = vertices
        # sort faces
        for i in range(len(faces)):
            f = faces[i]
            if f[0] > f[1]:
                f[0], f[1] = f[1], f[0]
            if f[0] > f[2]:
                f[0], f[2] = f[2], f[0]
            if f[1] > f[2]:
                f[1], f[2] = f[2], f[1]
            faces[i] = f
        self.faces = faces
        #initialize edge
        for e in self.faces:
            a, b, c = e[0], e[1], e[2]
            edg = np.array([[a,b], [a,c],[b,c]], dtype=np.int)
            self.edges = np.append(self.edges, edg)
        self.edges = np.reshape(self.edges, (-1, 2))
        self.edges = np.unique(self.edges, axis=0)
    
# TODO input mesh vertex and cells, return dual mesh

def dual_mesh(mesh):
    points = mesh.points
    faces = mesh.cells[0][1]
    msh = Mesh(points, faces)
    


def write_mesh(file_name='data/output.obj', mesh):
    vertcies = mesh.vertcies
    faces = mesh.faces
    file = open(file_name, mode='w')

if __name__ == '__main__':
    mesh = meshio.read('data/bunny.obj')
    #dm = dual_mesh(mesh)
    v = mesh.points
    f = mesh.cells[0][1]
    m = Mesh(v, f)
    

