from mesh_gen1 import refine_mesh, nodes2edge, nodes2element, edge2element
from solve import solve, triangle_bases, triangle_grads 
import numpy as np
coordinates = []
triangles = []
dirichlet =[]
neumann = []
quadrilaterals = []

all = [coordinates, triangles, dirichlet, neumann]
all_files = ['coordinate.dat', 'element.dat', 'Dirichlet.dat', 'Neumann.dat']

for i in range(len(all)):
    with open(all_files[i], 'r') as file:
        # Use a generator expression to filter out comment lines
        lines = (line for line in file if line.strip() and not line.strip().startswith('%'))

        # Parse each valid line into a tuple of integers
        if i == 0:
    
            all[i].extend([tuple(map(float, line.strip().split())) for line in lines])
        else:
            all[i].extend([tuple(map(int, line.strip().split())) for line in lines])

for i in range(len(triangles)):
    x1, y1 = coordinates[triangles[i][1]-1][1], coordinates[triangles[i][1]-1][2]
    x2, y2 = coordinates[triangles[i][2]-1][1], coordinates[triangles[i][2]-1][2]
    x3, y3 = coordinates[triangles[i][3]-1][1], coordinates[triangles[i][3]-1][2]
    lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2),
               np.sqrt((x3 - x2)**2 + (y3 - y2)**2),
               np.sqrt((x1 - x3)**2 + (y1 - y3)**2)]
    ind_hyp = np.argmax(lengths)
    # keep hypotenuse as between nodes 2 nd 3 by shifting circularly using np
    triangles[i] = (triangles[i][0], triangles[i][(ind_hyp + 2) % 3 + 1], triangles[i][(ind_hyp + 3) % 3 + 1], triangles[i][(ind_hyp + 4) % 3 + 1])

    
        
        

# Refine the mesh
def get_errors(coordinates, triangles,quadrilaterals, dirichlet, neumann, f , g, u_d) :
    coordinates, triangles, dirichlet, neumann = refine_mesh(coordinates, triangles, dirichlet, neumann, 3)
    u_h_vals = solve(coordinates, triangles, quadrilaterals, f, g, u_d, neumann, dirichlet)
    bases = triangle_bases(coordinates, triangles)
    u_h = lambda x, y: sum(u_h_vals[i] * bases[i](x, y) for i in range(len(bases)))
    # get error triangle_wise
    triangle_error = np.zeros(len(triangles))
    for i in range(len(triangles)):
        x1, y1 = coordinates[triangles[i][1]-1][1], coordinates[triangles[i][1]-1][2]
        x2, y2 = coordinates[triangles[i][2]-1][1], coordinates[triangles[i][2]-1][2]
        x3, y3 = coordinates[triangles[i][3]-1][1], coordinates[triangles[i][3]-1][2]
        h_t = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        median = (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3
        area = 0.5 * np.linalg.det(np.array([[1, 1, 1],
                                             [x1, x2, x3],
                                             [y1, y2, y3]]))
        triangle_error[i] = (f(median[0], median[1])**2 ) * area * h_t**2 / 3
        # nodes2edge_matrix = nodes2edge(coordinates, triangles)
        grads = triangle_grads(coordinates, triangles)
        grad_u = lambda x, y: (sum(grads[i][0](x, y) * u_h_vals[i] for i in range(len(grads))),
                                sum(grads[i][1](x, y) * u_h_vals[i] for i in range(len(grads))))
        for j in range(3) :
            x1, y1 = coordinates[triangles[i][j+1]-1][1], coordinates[triangles[i][j+1]-1][2]
            x2, y2 = coordinates[triangles[i][(j+2) % 3 + 1]-1][1], coordinates[triangles[i][(j+2) % 3 + 1]-1][2]
            edge_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            normal = np.array([- (y2 - y1) / edge_length, (x2 - x1) / edge_length])
            edge_midpoint = (x1 + x2) / 2, (y1 + y2) / 2
            triangle_error[i] += np.dot(grad_u(edge_midpoint[0], edge_midpoint[1]), normal) ** 2 * edge_length / 2
        
        triangle_error[i] = np.sqrt(triangle_error[i])
    triangle_wise_error = [(i + 1, triangle_error[i]) for i in range(len(triangle_error))]
    triangle_wise_error.sort(key=lambda x: x[1], reverse=True)
    return triangle_wise_error
    

def adaptive_refinement(coordinates, triangles, quadrilaterals, dirichlet, neumann, f, g, u_d, num_iters_unif):
    
def adaptive(coordinates, triangles, quadrilaterals, dirichlet, neumann, f, g, u_d, num_iters_unif):
    coordinates, triangles, dirichlet, neumann = refine_mesh(coordinates, triangles, dirichlet, neumann, num_iters_unif)
    errors = get_errors(coordinates, triangles, quadrilaterals, dirichlet, neumann, f, g, u_d)
    # do adaptive mesh refinement
    
    