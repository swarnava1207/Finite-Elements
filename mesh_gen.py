import numpy as np
import time
from datetime import datetime
from scipy.sparse import csr_matrix
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
    import matplotlib
    matplotlib.use('Agg')  # Use TkAgg backend for interactive plotting
    import matplotlib.pyplot as plt
    import numpy as np

    coordinates_np = np.array(coordinates)
    elements_np = np.array(elements) - 1  # Adjust for zero-based indexing in Python\
    plt.figure(figsize=(8, 8))
    for element in elements_np:
        pts = coordinates_np[element]
        plt.fill(pts[:, 0], pts[:, 1], edgecolor='black', fill=True)
    plt.scatter(coordinates_np[:, 0], coordinates_np[:, 1], color='red', s=10)
    plt.title('Mesh Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.grid()
    plt.savefig(f'mesh_visualization_{datetime.now()}.png')
    #plt.show()
    #print("Number of elements:", len(elements_np))

#show_mesh()

def nodes2element(coordinates, elements):
    

    num_nodes = len(coordinates)
    num_elements = len(elements)

    connectivity = np.zeros((num_nodes, num_nodes), dtype=int)
   
    for k,element in enumerate(elements):
        for i in range(4):
            node1 = element[i % 3]
            node2 = element[(i + 1) % 3]
            connectivity[node1 - 1, node2 - 1] = k + 1
            
    return csr_matrix(connectivity)

def nodes2edge(coordinates, elements):
    num_nodes = len(coordinates)
    element_matrix = nodes2element(coordinates, elements).toarray()
    
    # Create a boolean mask where either element_matrix[i,j] or element_matrix[j,i] is non-zero
    mask = (element_matrix != 0) | (element_matrix.T != 0)
    
    # Create a symmetric connectivity matrix
    connectivity = np.zeros((num_nodes, num_nodes), dtype=int)
    
    # Get upper triangular indices where mask is True
    rows, cols = np.triu_indices(num_nodes, k=1)
    valid_edges = mask[rows, cols]
    
    # Assign edge numbers (1-based indexing)
    edge_indices = np.where(valid_edges)[0]
    edge_count = len(edge_indices)
    
    # Assign edge numbers to both (i,j) and (j,i) to maintain symmetry
    connectivity_vals = np.arange(1, edge_count + 1)
    for idx, edge_num in zip(edge_indices, connectivity_vals):
        i, j = rows[idx], cols[idx]
        connectivity[i, j] = edge_num
        connectivity[j, i] = edge_num
    
    return csr_matrix(connectivity), edge_count

def edge2element(coordinates, elements):
    nodes2edge_matrix, edge_no = nodes2edge(coordinates, elements)
    nodes2edge_matrix = nodes2edge_matrix.toarray()
    nodes2element_matrix = nodes2element(coordinates, elements).toarray()
    
    # Initialize result matrix
    sol_matrix = np.zeros((edge_no, 4), dtype=int)
    
    # Get upper triangular indices where edges exist
    i_indices, j_indices = np.where(np.triu(nodes2edge_matrix > 0))
    edge_ids = nodes2edge_matrix[i_indices, j_indices] - 1  # 0-based indexing
    
    # Create masks for the two conditions
    condition = (nodes2element_matrix[i_indices, j_indices] == 0) | (nodes2element_matrix[j_indices, i_indices] != 0)
    
    # For condition True
    true_edges = edge_ids[condition]
    true_i = i_indices[condition]
    true_j = j_indices[condition]
    
    sol_matrix[true_edges, 0] = true_j + 1  # 1-based indexing
    sol_matrix[true_edges, 1] = true_i + 1
    sol_matrix[true_edges, 3] = nodes2element_matrix[true_i, true_j]
    sol_matrix[true_edges, 2] = nodes2element_matrix[true_j, true_i]
    
    # For condition False
    false_edges = edge_ids[~condition]
    false_i = i_indices[~condition]
    false_j = j_indices[~condition]
    
    sol_matrix[false_edges, 0] = false_i + 1
    sol_matrix[false_edges, 1] = false_j + 1
    sol_matrix[false_edges, 2] = nodes2element_matrix[false_i, false_j]
    sol_matrix[false_edges, 3] = nodes2element_matrix[false_j, false_i]
    
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
    
    nodes2element_matrix = nodes2element(coordinates, elements).toarray()
    nodes2edge_matrix, edge_no = nodes2edge(coordinates, elements)
    nodes2edge_matrix = nodes2edge_matrix.toarray()
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
    new_node2element_matrix = nodes2element(new_coordinates, new_elements).toarray()
    end = time.time()
    print(f"Time taken to compute node to element connectivity matrix: {end - start:.4f} seconds")
    start = time.time()
    new_node2edge_matrix, new_edge_no = nodes2edge(new_coordinates, new_elements)
    new_node2edge_matrix = new_node2edge_matrix.toarray()
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

coordinates, elements, dirichlet, neumann, node2element_matrix, node2edge_matrix, edge2element_matrix = refine_mesh(coordinates, elements, dirichlet, neumann, 5)
show_mesh(coordinates, elements)






    
        


    
