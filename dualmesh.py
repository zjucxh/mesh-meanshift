import meshio
import numpy as np

class Mesh:
    vertices = []
    edges = []
    faces = []
    adjacencyMatrix = []
    
    #TODO modify __init__ for obj file does not contain edges explicitly
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
        
    
# TODO input mesh vertex and cells, return dual mesh

def dual_mesh(mesh):
    points = mesh.points
    faces = mesh.cells[0][1]


def write_mesh(file_name, Mesh)
if __name__ == '__main__':
    mesh = meshio.read('data/bunny.obj')
    dm = dual_mesh(mesh)

