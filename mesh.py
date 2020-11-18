import numpy as np

class Mesh:
    Vertices = []
    Edges = []
    Faces = []
    NVertices = 0
    AdjacencyMatrix = []
    #TODO modify __init__ for obj file does not contain edges explicitly
    def __init__(self, vertices, edges, faces):
        self.Vertices = vertices
        self.Edges = edges
        self.Faces = faces
        self.NVertices = len(vertices)
        self.adjacency_matrix()

    def adjacency_matrix(self):
        self.AdjacencyMatrix = np.zeros(shape=(self.NVertices, self.NVertices),dtype=bool)
        for e in self.Edges:
            self.AdjacencyMatrix[e[0],e[1]] = True
            self.AdjacencyMatrix[e[1],e[0]] = True


    def neighbouring_vertices(self, idx):
        vertices = self.AdjacencyMatrix[idx,:]
        #return [i for i in range(len(vertices)) if vertices[i]==True]
        return np.where(vertices == True)
    def laplacian_of_graph(self, idx) :
        vi = self.Vertices[idx]
        neighbours_idx = self.neighbouring_vertices(idx)
        neighbours = [self.Vertices[e] for e in neighbours_idx][0]
        #print(neighbours)
        n = len(neighbours)
        center = 1.0/n * sum(neighbours)
        print(vi-center)
        return vi-center
def laplacian_optim():
    m1 = Mesh(np.array([[2,0,2], [0,2,2], [-2, 0, 2], [0,-2,2]], dtype=np.float), np.array([[0,1], [1,2], [0,2], [2,3], [0,3]], dtype=np.int), np.array([[0,1,2], [0,2,3]], dtype=np.int))
    m2 = Mesh(np.array([[4,0,2], [0,2,2], [-2, 0, 2], [0,-2,2]], dtype=np.float), np.array([[0,1], [1,2], [0,2], [2,3], [0,3]], dtype=np.int), np.array([[0,1,2], [0,2,3]], dtype=np.int))
    A = np.array([],dtype=np.float)
    B = np.array([],dtype=np.float)

    for idx in range(len(m1.Vertices)):
        a = m1.laplacian_of_graph(idx)
        #print('#############')
        #print(a)
        A = np.append(A,a, axis=0)
    A = np.reshape(A, (-1,3))
    print("A = ..")
    print(A)
    for idx in range(len(m2.Vertices)):
        b = m2.laplacian_of_graph(idx)
        B = np.append(B, b, axis=0)
    B = np.reshape(B, (-1, 3))
    A = A[:,0:2]
    B = B[:,0:2]
    print("B = .. ")
    print(B)
    return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)),A.T),B)





#m1 = Mesh(np.array([[2,0,0], [0,2,0], [-2, 0, 0], [0,-2,0]], dtype=np.float), np.array([[0,1], [1,2], [0,2], [2,3], [0,3]], dtype=np.int), np.array([[0,1,2], [0,2,3]], dtype=np.int))

#x = laplacian_optim()
#print(x)
