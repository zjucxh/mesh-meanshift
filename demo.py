import meshio
import dualmesh as dm

import matplotlib.pyplot as plt

mesh = meshio.read('./data/bunny.obj')
points = mesh.points
faces = mesh.cells[0][1]
dual_mesh = dm.dual_mesh(mesh)
print(cells)
#meshio.write('data/dual-bunny.obj', dual_mesh)
