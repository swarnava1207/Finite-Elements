import numpy as np
import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg

import time
from datetime import datetime
from scipy.sparse import csr_matrix, lil_matrix, triu, coo_matrix
import matplotlib
#matplotlib.use('Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import numpy as np
coordinates = []
elements = []
dirichlet =[]
neumann = []

all = [coordinates, elements, dirichlet, neumann]
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

def show_mesh(coordinates, elements):
    from vispy import use
    use('glfw')
    from vispy import scene, color, app
    coords = np.asarray(coordinates, dtype=np.float32)
    elems  = np.asarray(elements,   dtype=np.int32) - 1
    ntri   = elems.shape[0]

    # Pick per-triangle colors
    cmap = color.get_colormap("turbo")
    tri_colors = cmap.map(np.linspace(0, 1, ntri))

    # Build canvas + 2D pan/zoom view
    canvas = scene.SceneCanvas(keys='interactive',
                               bgcolor='white',
                               size=(800,800),
                               show=True)
    view   = canvas.central_widget.add_view()
    view.camera = 'panzoom'

    # Draw filled triangles (no edges)
    mesh = scene.visuals.Mesh(
        vertices=coords,
        faces=elems,
        face_colors=tri_colors,
        shading=None        # flat shading
    )
    view.add(mesh)

    all_edges = np.vstack([
    coords[elems[:, [0,1]]],
    coords[elems[:, [1,2]]],
    coords[elems[:, [2,0]]],
    ]).reshape(-1, 2, 2)

    # Flatten to a (N_points, 2) array for Line
    segs = all_edges.reshape(-1, 2)

    # Create a Line visual, in 'segments' mode, colored black
    lines = scene.visuals.Line(
        pos=segs,
        connect='segments',
        color=(0, 0, 0, 1),   # RGBA tuple is more reliable than 'black'
        width=1.0,
        method='gl'           # ensure we’re using the GL backend
    )
    # Draw on top of the mesh:
    # Disable depth‐testing so edges aren’t hidden by the fills:
    lines.set_gl_state(depth_test=False)

    # Force lines to draw *after* the mesh:
    lines.order = 1                # give it a higher draw-order

    view.add(lines)

    # Draw nodes on top
    scatter = scene.visuals.Markers()
    scatter.set_data(coords, face_color='red', size=5)
    view.add(scatter)

    # Fit the data
    view.camera.set_range(x=(coords[:,0].min(), coords[:,0].max()),
                          y=(coords[:,1].min(), coords[:,1].max()))

    app.run()


#show_mesh()

def nodes2element(coordinates, elements):
    num_nodes = len(coordinates)
    num_elements = len(elements)

    connectivity = lil_matrix((num_nodes, num_nodes), dtype=np.int32)
   
    for k,element in enumerate(elements):
        for i in range(4):
            node1 = element[i % 3]
            node2 = element[(i + 1) % 3]
            connectivity[node1 - 1, node2 - 1] = k + 1
            
    return csr_matrix(connectivity)

def nodes2edge(coordinates, elements):
    # Get the CSR element-incidence matrix
    element_matrix = nodes2element(coordinates, elements)
    num_nodes = element_matrix.shape[0]

    # Build symmetric adjacency: nonzero if i->j or j->i
    sym = element_matrix + element_matrix.T

    # Extract strict upper-triangular nonzero entries
    upper = triu(sym, k=1).tocoo()
    rows, cols = upper.row, upper.col

    # Number of edges and their 1-based IDs
    edge_count = len(rows)
    edge_ids = np.arange(1, edge_count + 1)

    # Build symmetric connectivity entries: both (i,j) and (j,i)
    row_inds = np.concatenate([rows, cols])
    col_inds = np.concatenate([cols, rows])
    data     = np.concatenate([edge_ids, edge_ids])

    connectivity = coo_matrix(
        (data, (row_inds, col_inds)),
        shape=(num_nodes, num_nodes)
    ).tocsr()

    return connectivity, edge_count



def edge2element(coordinates, elements):
    # nodes2edge and nodes2element now return csr_matrix
    nodes2edge_matrix, edge_no = nodes2edge(coordinates, elements)      # csr_matrix, and total # of edges
    nodes2element_matrix = nodes2element(coordinates, elements)         # csr_matrix
    # print(f"Number of edges: {edge_no}")
    # print(f"nodes2edge_matrix: {nodes2edge_matrix}")
    # print(f"nodes2element_matrix: {nodes2element_matrix}")
    # Extract only the strict upper‐triangular entries of the edge‐connectivity
    upper = triu(nodes2edge_matrix, k=1).tocoo()
    i_indices = upper.row
    j_indices = upper.col
    edge_ids  = upper.data.astype(int) - 1    # convert to 0‐based
    # print(f"i_indices: {i_indices}, j_indices: {j_indices}, edge_ids: {edge_ids}")
    # Look up the two element‐incidence values for each (i,j)
    # Missing entries in a CSR default to 0
    v_ij = np.array(
        [nodes2element_matrix[ii, jj] for ii, jj in zip(i_indices, j_indices)],
        dtype=int
    )
    v_ji = np.array(
        [nodes2element_matrix[jj, ii] for ii, jj in zip(i_indices, j_indices)],
        dtype=int
    )

    # The same "swap" condition as before
    condition = (v_ij == 0) | (v_ji != 0)

    # Prepare the output
    sol_matrix = np.zeros((edge_no, 4), dtype=int)

    # Apply the True‐case assignments
    tmask = condition
    te = edge_ids[tmask]
    ti = i_indices[tmask]
    tj = j_indices[tmask]
    # print(f"te: {te}, ti: {ti}, tj: {tj}")
    sol_matrix[ te, 0] = tj + 1
    sol_matrix[ te, 1] = ti + 1
    sol_matrix[ te, 3] = v_ij[tmask]
    sol_matrix[ te, 2] = v_ji[tmask]

    # Apply the False‐case assignments
    fmask = ~condition
    fe = edge_ids[fmask]
    fi = i_indices[fmask]
    fj = j_indices[fmask]
    # print(f"fe: {fe}, fi: {fi}, fj: {fj}")
    sol_matrix[ fe, 0] = fi + 1
    sol_matrix[ fe, 1] = fj + 1
    sol_matrix[ fe, 2] = v_ij[fmask]
    sol_matrix[ fe, 3] = v_ji[fmask]

    return sol_matrix


print("Nodes to Element Connectivity Matrix:")
element_matrix = nodes2element(coordinates, elements)
print(element_matrix.toarray())

print("Nodes to Edge Connectivity Matrix:")
edge_matrix, edge_no = nodes2edge(coordinates, elements)
print(edge_matrix.toarray())
            
print("Edge to Element Connectivity Matrix:")
edge_element_matrix = edge2element(coordinates, elements)
print(edge_element_matrix)


                
def add_nodes(coordinates, elements, dirichlet, neumann):
    
    #nodes2element_matrix = nodes2element(coordinates, elements).toarray()
    nodes2edge_matrix, edge_no = nodes2edge(coordinates, elements)
    #nodes2edge_matrix = nodes2edge_matrix.toarray()
    edge2element_matrix = edge2element(coordinates, elements)

    num_nodes = len(coordinates)
    num_elements = len(elements)

    new_nodes = []
    new_elements = []
    new_dirichlet = []
    new_neumann = []
    new_edges = []
    start = time.time()
    for i in range(edge_no):
        edge = edge2element_matrix[i]
        node1_coords = coordinates[edge[0] - 1]
        node2_coords = coordinates[edge[1] - 1]
        new_node_coords = ((node1_coords[0] + node2_coords[0]) / 2,
                           (node1_coords[1] + node2_coords[1]) / 2)
        new_nodes.append(new_node_coords)


    for element in elements :
        node1, node2, node3 = element
        edge1 = nodes2edge_matrix[node1 - 1, node2 - 1]
        edge2 = nodes2edge_matrix[node2 - 1, node3 - 1]
        edge3 = nodes2edge_matrix[node3 - 1, node1 - 1]
        new_node_1_index = num_nodes + edge1
        new_node_2_index = num_nodes + edge2
        new_node_3_index = num_nodes + edge3 
        new_elements.extend([(new_node_1_index, new_node_2_index, new_node_3_index),
                            (new_node_1_index, node2, new_node_2_index),
                            (new_node_2_index, node3, new_node_3_index),
                            (new_node_3_index, node1, new_node_1_index)])
    for node in dirichlet:
        edge = nodes2edge_matrix[node[0] - 1, node[1] - 1]
        new_node = new_nodes[edge - 1]
        new_node_index = num_nodes + edge
        new_dirichlet.append((new_node_index, node[0]))
        new_dirichlet.append((new_node_index, node[1]))
    
    for node in neumann:
        edge = nodes2edge_matrix[node[0] - 1, node[1] - 1]
        new_node = new_nodes[edge - 1]
        new_node_index = num_nodes + edge
        new_neumann.append((new_node_index, node[0]))
        new_neumann.append((new_node_index, node[1]))
    end = time.time()
    print(f"Time taken to add nodes and elements: {end - start:.4f} seconds")
    new_coordinates = coordinates + new_nodes
    start = time.time()
    new_node2element_matrix = nodes2element(new_coordinates, new_elements)#.toarray()
    end = time.time()
    print(f"Time taken to compute node to element connectivity matrix: {end - start:.4f} seconds")
    start = time.time()
    new_node2edge_matrix, new_edge_no = nodes2edge(new_coordinates, new_elements)
    #new_node2edge_matrix = new_node2edge_matrix.toarray()
    end = time.time()
    print(f"Time taken to compute node to edge connectivity matrix: {end - start:.4f} seconds")
    start = time.time()
    new_edge2element_matrix = edge2element(new_coordinates, new_elements)
    end = time.time()
    print(f"Time taken to compute edge to element connectivity matrices: {end - start:.4f} seconds")
    print(f"Number of nodes: {len(new_coordinates)}")
    print(f"Number of elements: {len(new_elements)}")
    return (new_coordinates, new_elements, new_dirichlet, new_neumann,
            new_node2element_matrix, new_node2edge_matrix, new_edge2element_matrix)

def refine_mesh(coordinates, elements, dirichlet, neumann, iterations):
    iter = 0
    while iter < iterations:
        print(f"Iteration {iter + 1} of {iterations}")
        coordinates, elements, dirichlet, neumann, node2element_matrix, node2edge_matrix, edge2element_matrix = add_nodes(coordinates, elements, dirichlet, neumann)
        iter += 1
    return coordinates, elements, dirichlet, neumann, node2element_matrix, node2edge_matrix, edge2element_matrix

coordinates, elements, dirichlet, neumann, node2element_matrix, node2edge_matrix, edge2element_matrix = refine_mesh(coordinates, elements, dirichlet, neumann, 10)
show_mesh(coordinates, elements)






    
        


    