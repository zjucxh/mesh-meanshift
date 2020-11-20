import numpy as np

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

