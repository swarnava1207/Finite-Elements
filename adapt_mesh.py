from mesh_gen1 import refine_mesh, nodes2edge, nodes2element, edge2element, show_mesh
from solve import solve, triangle_grads
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

# coordinates = [(i + 1, coord[0], coord[1]) for i, coord in enumerate(coordinates)]

for i in range(len(triangles)):
    x1, y1 = coordinates[triangles[i][1]-1][1], coordinates[triangles[i][1]-1][2]
    x2, y2 = coordinates[triangles[i][2]-1][1], coordinates[triangles[i][2]-1][2]
    #print("coordinates:", coordinates[triangles[i][3]-1])
    x3, y3 = coordinates[triangles[i][3]-1][1], coordinates[triangles[i][3]-1][2]
    lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2),
               np.sqrt((x3 - x2)**2 + (y3 - y2)**2),
               np.sqrt((x1 - x3)**2 + (y1 - y3)**2)]
    ind_hyp = np.argmax(lengths)
    # keep hypotenuse as between nodes 2 nd 3 by shifting circularly using np
    triangles[i] = (triangles[i][0], triangles[i][(ind_hyp + 2) % 3 + 1], triangles[i][(ind_hyp + 3) % 3 + 1], triangles[i][(ind_hyp + 4) % 3 + 1])

coordinates = np.array(coordinates)
triangles = np.array(triangles)
dirichlet = np.array(dirichlet)
neumann = np.array(neumann)
        

# Refine the mesh
# def get_errors(coordinates, triangles,quadrilaterals, dirichlet, neumann, f , g, u_d) :
#     #coordinates, triangles, dirichlet, neumann = refine_mesh(coordinates, triangles, dirichlet, neumann, 3)
#     u_h_vals = solve(coordinates, triangles, quadrilaterals, f, g, u_d, neumann, dirichlet)
#     bases = triangle_bases(coordinates, triangles)
#     u_h = lambda x, y: sum(u_h_vals[i] * bases[i](x, y) for i in range(len(bases)))
#     # get error triangle_wise
#     triangle_error = np.zeros(len(triangles))
#     for i in range(len(triangles)):
#         x1, y1 = coordinates[triangles[i][1]-1][1], coordinates[triangles[i][1]-1][2]
#         x2, y2 = coordinates[triangles[i][2]-1][1], coordinates[triangles[i][2]-1][2]
#         x3, y3 = coordinates[triangles[i][3]-1][1], coordinates[triangles[i][3]-1][2]
#         h_t = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
#         median = (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3
#         area = 0.5 * np.linalg.det(np.array([[1, 1, 1],
#                                              [x1, x2, x3],
#                                              [y1, y2, y3]]))
#         triangle_error[i] = (f(median[0], median[1])**2 ) * area * h_t**2 / 3
#         # nodes2edge_matrix = nodes2edge(coordinates, triangles)
#         grads = triangle_grads(coordinates, triangles)
#         grad_u = lambda x, y: (sum(grads[i][0](x, y) * u_h_vals[i] for i in range(len(grads))),
#                                 sum(grads[i][1](x, y) * u_h_vals[i] for i in range(len(grads))))
#         for j in range(3) :
#             x1, y1 = coordinates[triangles[i][j+1]-1][1], coordinates[triangles[i][j+1]-1][2]
#             x2, y2 = coordinates[triangles[i][(j+2) % 3 + 1]-1][1], coordinates[triangles[i][(j+2) % 3 + 1]-1][2]
#             edge_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#             normal = np.array([- (y2 - y1) / edge_length, (x2 - x1) / edge_length])
#             edge_midpoint = (x1 + x2) / 2, (y1 + y2) / 2
#             triangle_error[i] += np.dot(grad_u(edge_midpoint[0], edge_midpoint[1]), normal) ** 2 * edge_length / 2
        
#         triangle_error[i] = np.sqrt(triangle_error[i])
#     triangle_wise_error = [(i + 1, triangle_error[i]) for i in range(len(triangle_error))]
#     triangle_wise_error.sort(key=lambda x: x[1], reverse=True)
#     return triangle_wise_error

def error_tri(coordinates, triangles, f) :
    errors = np.zeros(len(triangles))
    for i in range(len(triangles)):
        nodes = np.array(triangles[i][1:]) - 1
        p1, p2, p3 = tuple(coordinates[nodes])
        area = 0.5 * np.linalg.det(np.array([[1, 1, 1],
                                             [p1[1], p2[1], p3[1]],
                                             [p1[2], p2[2], p3[2]]]))
        mp23 = (p2[1] + p3[1]) / 2, (p2[2] + p3[2]) / 2
        mp12 = (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2
        mp31 = (p3[1] + p1[1]) / 2, (p3[2] + p1[2]) / 2
        error = (f(mp23)**2 + f(mp12)**2 + f(mp31)**2) * area / 3
        errors[i] = np.sqrt(error * area)
    # errors = [(i + 1, errors[i]) for i in range(len(errors))]
    # errors.sort(key=lambda x: x[1], reverse=True)
    return errors

def error_edge(coordinates, triangles, f, g, neumann, n2ed, ed2el, grads, u_h) :
    num_edges = len(ed2el)
    errors = np.zeros(len(triangles))
    for i in range(num_edges):
        element1 = ed2el[i][2]
        element2 = ed2el[i][3]
        if element2 != 0 :
            node1 = ed2el[i][0] - 1
            node2 = ed2el[i][1] - 1
            coord1, coord2 = coordinates[node1], coordinates[node2]
            he = np.sqrt((coord2[1] - coord1[1])**2 + (coord2[2] - coord1[2])**2)
            normal = np.array([- (coord2[2] - coord1[2]) / he, (coord2[1] - coord1[1]) / he])
            nodes1 = triangles[element1 - 1][1:]
            # Get the corresponding nodal solution values from u_h
            u_nodal1 = u_h[nodes1 - 1]
            # Get the three basis gradient functions for this element
            basis_grad_funcs1 = grads[element1 - 1]
            # Compute the gradient of uh by summing the weighted basis gradients
            grad_uh1 = (u_nodal1[0] * basis_grad_funcs1[0](0,0) +
                        u_nodal1[1] * basis_grad_funcs1[1](0,0) +
                        u_nodal1[2] * basis_grad_funcs1[2](0,0))

            # --- Gradient of uh in element2 ---
            # Get the three 1-based node numbers for the second element
            nodes2 = triangles[element2 - 1][1:]
            # Get the corresponding nodal solution values from u_h
            u_nodal2 = u_h[nodes2 - 1]
            # Get the three basis gradient functions for this element
            basis_grad_funcs2 = grads[element2 - 1]
            # Compute the gradient of uh by summing the weighted basis gradients
            grad_uh2 = (u_nodal2[0] * basis_grad_funcs2[0](0,0) +
                        u_nodal2[1] * basis_grad_funcs2[1](0,0) +
                        u_nodal2[2] * basis_grad_funcs2[2](0,0))
            grad_diff = grad_uh1 - grad_uh2
            err = (he ** 2) * np.dot(grad_diff, normal) ** 2 / 2
            errors[element1 - 1] += err
            errors[element2 - 1] += err
    for i in range(len(neumann)) :
        node1 = neumann[i][1] - 1
        node2 = neumann[i][2] - 1
        element = ed2el[n2ed[node1][node2]][2]
        coord1, coord2 = coordinates[node1], coordinates[node2]
        he = np.sqrt((coord2[1] - coord1[1])**2 + (coord2[2] - coord1[2])**2)
        mp = (coord1[1] + coord2[1]) / 2, (coord1[2] + coord2[2]) / 2
        normal= np.array([- (coord2[2] - coord1[2]) / he, (coord2[1] - coord1[1]) / he])
        nodes1 = triangles[element1 - 1][1:]
        # Get the corresponding nodal solution values from u_h
        u_nodal1 = u_h[nodes1 - 1]
        # Get the three basis gradient functions for this element
        basis_grad_funcs1 = grads[element1 - 1]
        # Compute the gradient of uh by summing the weighted basis gradients
        grad = (u_nodal1[0] * basis_grad_funcs1[0](0,0) +
                    u_nodal1[1] * basis_grad_funcs1[1](0,0) +
                    u_nodal1[2] * basis_grad_funcs1[2](0,0))
        err = (he ** 2) * (np.dot(grad, normal) - g(mp)) ** 2 / 2
        errors[element - 1] += err
    errors = np.sqrt(errors)
    # errors = [(i + 1, np.sqrt(errors[i])) for i in range(len(errors))]
    # errors.sort(key=lambda x: x[1], reverse=True)
    return errors



def adaptive_refinement(coordinates, triangles, dirichlet, neumann, error_triangle, theta, n2el, n2ed, num_edges):
    # error = error_tri(coordinates, triangles, f) + error_edge(coordinates, triangles, f, g, neumann, n2ed, edge2element(n2ed, num_edges), triangle_grads(coordinates, triangles))
    num_triangles = len(triangles)
    total_error = sum(err[1] for err in error_triangle)
    cur_error = 0
    marker = np.zeros(num_edges, dtype=int)
    for i in range(num_triangles):
        if cur_error / total_error > theta:
            break
        index = 1
        ct = i
        while index == 1 :
            longest_edge = n2ed(triangles[ct][2] - 1, triangles[ct][3] - 1) - 1
            if marker[longest_edge] > 0 :
                index = 0
            else :
                num_coords = len(coordinates) + 1
                marker[longest_edge] = num_coords
                x1, y1 = coordinates[triangles[ct][2] - 1][1], coordinates[triangles[ct][2] - 1][2]
                x2, y2 = coordinates[triangles[ct][3] - 1][1], coordinates[triangles[ct][3] - 1][2]
                new_coord = (x1 + x2) / 2, (y1 + y2) / 2
                coordinates = np.append(coordinates, [new_coord], axis=0)
                cur_error += error_triangle[ct][1]
                ct = n2el[triangles[ct][3] - 1][triangles[ct][2] - 1]
                if ct == 0 :
                    index = 0
        
    for i in range(num_triangles):
        longest_edge = n2ed(triangles[i][2] - 1, triangles[i][3] - 1) - 1
        if marker[longest_edge] > 0:
            triangles[i] = (triangles[i][0], marker[longest_edge], triangles[i][3], triangles[i][1])
            triangles.append((len(triangles) + 1, marker[longest_edge], triangles[i][1], triangles[i][2]))
            left = n2ed(triangles[i][1] - 1, triangles[i][2] - 1) - 1
            right = n2ed(triangles[i][3] - 1, triangles[i][1] - 1) - 1
            if marker[left] > 0:
                triangles[i] = (triangles[i][0], marker[left], marker[longest_edge], triangles[i][1])
                triangles.append((len(triangles) + 1, marker[left], triangles[i][2], marker[longest_edge]))
            if marker[right] > 0:
                triangles[-1] = (triangles[-1][0], marker[right], triangles[-1][1], marker[longest_edge])
                triangles.append((len(triangles) + 1, marker[right], marker[longest_edge], triangles[i][3]))
    
    for i in range(len(neumann)):
        edge = n2ed[neumann[i][1] - 1][neumann[i][2] - 1]
        if marker[edge] > 0:
            neumann[i] = (neumann[i][0], neumann[i][1], marker[edge])
            neumann.append((len(neumann) + 1, marker[edge], neumann[i][2]))
    for i in range(len(dirichlet)):
        edge = n2ed[dirichlet[i][1] - 1][dirichlet[i][2] - 1]
        if marker[edge] > 0:
            dirichlet[i] = (dirichlet[i][0], dirichlet[i][1], marker[edge])
            dirichlet.append((len(dirichlet) + 1, marker[edge], dirichlet[i][2]))
    coordinates = np.array(coordinates)
    triangles = np.array(triangles)
    dirichlet = np.array(dirichlet)
    neumann = np.array(neumann)
    return coordinates, triangles, dirichlet, neumann
    
def adaptive(coordinates, triangles, quadrilaterals, dirichlet, neumann,u_d, f, g, num_iters_unif, num_iter_adapt, theta):
    coordinates, triangles, dirichlet, neumann = refine_mesh(coordinates, triangles, dirichlet, neumann, num_iters_unif)
    for _ in range(num_iter_adapt):
        errors_triangle = error_tri(coordinates, triangles, f)
        grads = triangle_grads(coordinates, triangles)
        u_h = solve(coordinates, triangles, quadrilaterals, u_d, f, g, dirichlet, neumann)
        n2ed, num_edges = nodes2edge(coordinates, triangles)
        n2el = nodes2element(coordinates, triangles)
        ed2el = edge2element(coordinates, triangles)
        error_edges = error_edge(coordinates, triangles, f, g, neumann, n2ed, ed2el, grads, u_h)
        error_triangle = [(i + 1, errors_triangle[i]) for i in range(len(errors_triangle))]
        error_triangle.sort(key=lambda x: x[1], reverse=True)
        coordinates, triangles, dirichlet, neumann = adaptive_refinement(coordinates, triangles, dirichlet, neumann, error_triangle, theta, n2el, n2ed, num_edges)

show_mesh(coordinates, triangles)
coordinates, triangles, dirichlet, neumann = adaptive(coordinates, triangles, quadrilaterals, dirichlet, neumann, lambda x, y: 0, lambda x, y: 1, lambda x, y: 0, 1, 1, 0.3)
show_mesh(coordinates, triangles)
