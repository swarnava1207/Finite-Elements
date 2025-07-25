import numpy as np
from scipy.sparse import csr_matrix,lil_matrix

coordinates = []
triangles = []
quadrilaterals = []
neumann = []
dirichlet = []



with open("coordinates.dat", 'r') as file:
    # Use a generator expression to filter out comment lines
    lines = (line for line in file if line.strip() and not line.strip().startswith('%'))

    # Parse each valid line into a tuple of integers
    coordinates = [tuple(map(float, line.strip().split())) for line in lines]


with open("elements3.dat", 'r') as file:
        # Use a generator expression to filter out comment lines
        lines = (line for line in file if line.strip() and not line.strip().startswith('%'))

        # Parse each valid line into a tuple of integers
        triangles = [tuple(map(int, line.strip().split())) for line in lines]

with open("elements4.dat", 'r') as file:
    # Use a generator expression to filter out comment lines
    lines = (line for line in file if line.strip() and not line.strip().startswith('%'))

    # Parse each valid line into a tuple of integers
    quadrilaterals = [tuple(map(int, line.strip().split())) for line in lines]

with open("neumann.dat", 'r') as file:
    # Use a generator expression to filter out comment lines
    lines = (line for line in file if line.strip() and not line.strip().startswith('%'))

    # Parse each valid line into a tuple of integers
    neumann = [tuple(map(int, line.strip().split())) for line in lines]

with open("dirichlet.dat", 'r') as file:
    # Use a generator expression to filter out comment lines
    lines = (line for line in file if line.strip() and not line.strip().startswith('%'))

    # Parse each valid line into a tuple of integers
    dirichlet = [tuple(map(int, line.strip().split())) for line in lines]

unique_dirichlet_nodes = set()
for i in range(len(dirichlet)):
    unique_dirichlet_nodes.add(dirichlet[i][1]-1)
    unique_dirichlet_nodes.add(dirichlet[i][2]-1)
unique_dirichlet_nodes = list(unique_dirichlet_nodes)
free_nodes = [i for i in range(len(coordinates)) if i not in unique_dirichlet_nodes]

def triangle_bases(coordinates, triangles):
    bases = np.zeros(len(coordinates), dtype=object)
    for tri in triangles:
        for i in range(1, 4):
            x1, y1 = coordinates[tri[i]-1][1], coordinates[tri[i]-1][2]
            x2, y2 = coordinates[tri[(i % 3) + 1]-1][1], coordinates[tri[(i % 3) + 1]-1][2]
            x3, y3 = coordinates[tri[(i + 1) % 3 + 1]-1][1], coordinates[tri[(i + 1) % 3 + 1]-1][2]
            # Define the numerator and denominator for the barycentric coordinates
            numer = lambda x, y: np.array([1 , x , y],
                                          [1 , x2 , y2],
                                          [1 , x3 , y3])
            denom = lambda x, y: np.array([1 , x1 , y1],
                                          [1 , x2 , y2],
                                          [1 , x3 , y3])
            bases[triangles[i]] = lambda x, y: np.linalg.det(numer(x, y)) / np.linalg.det(denom(x, y))
    
    return bases

def triangle_grads(coordinates, triangles):
    grads = np.zeros((len(triangles), 3), dtype=object)
    for tri in triangles:
        for i in range(1, 4):
            x1, y1 = coordinates[tri[i]-1][1], coordinates[tri[i]-1][2]
            x2, y2 = coordinates[tri[(i % 3) + 1]-1][1], coordinates[tri[(i % 3) + 1]-1][2]
            x3, y3 = coordinates[tri[(i + 1) % 3 + 1]-1][1], coordinates[tri[(i + 1) % 3 + 1]-1][2]
            
            # Define the gradient of the barycentric coordinates
            area = ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            grad_x = lambda x, y: (y2 - y3) / area
            grad_y = lambda x, y: (x3 - x2) / area
            grads[tri[0] - 1, i - 1] = (lambda x, y: np.array([grad_x(x, y), grad_y(x, y)]))
    
    return grads





def stiffness_matrices_tri(coordinates, triangles):
    # bases = triangle_bases(coordinates, triangles)
    # grads = triangle_grads(coordinates, triangles)
    # print("Coordinates:", coordinates)
    # print("Triangles:", triangles)
    stiffness_matrices = []
    for i in range(len(triangles)):
        #print(f"Triangle {i+1}: {triangles[i]}")
        x1, y1 = coordinates[triangles[i][1]-1][1], coordinates[triangles[i][1]-1][2]
        x2, y2 = coordinates[triangles[i][2]-1][1], coordinates[triangles[i][2]-1][2]
        x3, y3 = coordinates[triangles[i][3]-1][1], coordinates[triangles[i][3]-1][2]
        # print(f"Triangle {i+1}: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})")

        G = np.linalg.inv(np.array([[1, 1, 1],
                                    [x1, x2, x3],
                                    [y1, y2, y3]]))
        G_prime = G @ np.array([[0, 0],
                                [1, 0],
                                [0, 1]])
        #print(f"G_prime for triangle {i+1}:", G_prime)
        area = 0.5 * np.linalg.det(np.array([[1, 1, 1],
                                             [x1, x2, x3],
                                             [y1, y2, y3]]))
        #print(f"Area for triangle {i+1}:", area * 2)
        stiffness_matrix = G_prime @ G_prime.T * area
        stiffness_matrices.append(stiffness_matrix)
    
    return stiffness_matrices

def affines(coordinates,quadrilaterals):
    affines = []
    for quad in quadrilaterals:
        x1, y1 = coordinates[quad[1]-1][1], coordinates[quad[1]-1][2]
        x2, y2 = coordinates[quad[2]-1][1], coordinates[quad[2]-1][2]
        x3, y3 = coordinates[quad[3]-1][1], coordinates[quad[3]-1][2]
        x4, y4 = coordinates[quad[4]-1][1], coordinates[quad[4]-1][2]
        
        # Define the affine transformation
        A = np.array([[x2 - x1, x4 - x1],
                      [y2 - y1, y4 - y1]])
        b = np.array([x1, y1])
        
        affine = lambda e, p : A @ np.array([e, p]) + b
        affines.append(affine)
    return affines

def affines_totalderivative(coordinates, quadrilaterals):
    #affines = affines(coordinates, quadrilaterals)
    total_derivatives = []
    for i in range(len(quadrilaterals)):
        x1, y1 = coordinates[quadrilaterals[i][1]-1][1], coordinates[quadrilaterals[i][1]-1][2]
        x2, y2 = coordinates[quadrilaterals[i][2]-1][1], coordinates[quadrilaterals[i][2]-1][2]
        x3, y3 = coordinates[quadrilaterals[i][3]-1][1], coordinates[quadrilaterals[i][3]-1][2]
        x4, y4 = coordinates[quadrilaterals[i][4]-1][1], coordinates[quadrilaterals[i][4]-1][2]
        
        # Define the total derivative of the affine transformation
        A = np.array([[x2 - x1, x4 - x1],
                      [y2 - y1, y4 - y1]])
        
        total_derivative = A
        total_derivatives.append(total_derivative)
    
    return total_derivatives

def stiffness_matrices_quad(coordinates, quadrilaterals):
    #affines = affines(coordinates, quadrilaterals)
    total_derivatives = affines_totalderivative(coordinates, quadrilaterals)
    
    stiffness_matrices = []
    for i in range(len(quadrilaterals)):
        D_phi = total_derivatives[i]
        #a , b, c = np.linalg.inv(D_phi.T @ D_phi)(0, 0), np.linalg.inv(D_phi.T @ D_phi)(0, 1), np.linalg.inv(D_phi.T @ D_phi)(1, 1)
        B = np.linalg.inv(D_phi.T @ D_phi)

        C1 = np.array([[2, -2], [-2, 2]]) * B[0, 0] + np.array([[3, 0], [0, -3]]) * B[0, 1] + np.array([[2, 1], [1, 2]]) * B[1, 1]
        C2 = np.array([[-1, 1], [1, -1]]) * B[0, 0] + np.array([[-3, 0], [0, 3]]) * B[0, 1] + np.array([[-1, -2], [-2, -1]]) * B[1, 1]
        M = np.linalg.det(D_phi) * np.block([[C1, C2], [C2, C1]]) / 6
        stiffness_matrices.append(M)

    return stiffness_matrices

def assemble_stiffness_matrix(coordinates, triangles, quadrilaterals):
    A = lil_matrix((len(coordinates), len(coordinates)), dtype=np.float64)
    
    for i in range(len(triangles)):
        # x1, y1 = coordinates[triangles[i][1]-1][1], coordinates[triangles[i][1]-1][2]
        # x2, y2 = coordinates[triangles[i][2]-1][1], coordinates[triangles[i][2]-1][2]
        # x3, y3 = coordinates[triangles[i][3]-1][1], coordinates[triangles[i][3]-1][2]
        # print(f"Triangle {i+1}: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3})")
        stiffness_matrices = stiffness_matrices_tri(coordinates, triangles)
        #print("Stiffness matrix for triangle:", stiffness_matrices[i])
        for j in range(1,4):
            for k in range(1, 4):
                A[triangles[i][j]-1, triangles[i][k]-1] += stiffness_matrices[i][j-1 , k-1]
    #print("Stiffness matrix after triangles:", A[free_nodes, :][:, free_nodes])
    for i in range(len(quadrilaterals)):
        stiffness_matrices = stiffness_matrices_quad(coordinates, quadrilaterals)
        for j in range(1, 5):
            for k in range(1, 5):
                A[quadrilaterals[i][j]-1, quadrilaterals[i][k]-1] += stiffness_matrices[i][j-1, k-1]
    #print("Stiffness matrix after quadrilaterals:", A[free_nodes, :][:, free_nodes])
    A = csr_matrix(A)
    return A

def assemble_load_vector(coordinates, triangles, quadrilaterals, f, g, u_d, neumann, dirichlet):
    b = np.zeros(len(coordinates), dtype=np.float64)
    for i in range(len(triangles)):
        x1, y1 = coordinates[triangles[i][1]-1][1], coordinates[triangles[i][1]-1][2]
        x2, y2 = coordinates[triangles[i][2]-1][1], coordinates[triangles[i][2]-1][2]
        x3, y3 = coordinates[triangles[i][3]-1][1], coordinates[triangles[i][3]-1][2]
        area = 0.5 * np.linalg.det(np.array([[1, 1, 1],
                                             [x1, x2, x3],
                                             [y1, y2, y3]]))
        xs, ys = (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3
        b[triangles[i][1]-1] += f(xs, ys) * area / 3
        b[triangles[i][2]-1] += f(xs, ys) * area / 3
        b[triangles[i][3]-1] += f(xs, ys) * area / 3
        

    for i in range(len(quadrilaterals)):
        x1, y1 = coordinates[quadrilaterals[i][1]-1][1], coordinates[quadrilaterals[i][1]-1][2]
        x2, y2 = coordinates[quadrilaterals[i][2]-1][1], coordinates[quadrilaterals[i][2]-1][2]
        x3, y3 = coordinates[quadrilaterals[i][3]-1][1], coordinates[quadrilaterals[i][3]-1][2]
        x4, y4 = coordinates[quadrilaterals[i][4]-1][1], coordinates[quadrilaterals[i][4]-1][2]
        area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        xs, ys = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
        b[quadrilaterals[i][1]-1] += f(xs, ys) * area / 4
        b[quadrilaterals[i][2]-1] += f(xs, ys) * area / 4
        b[quadrilaterals[i][3]-1] += f(xs, ys) * area / 4
        b[quadrilaterals[i][4]-1] += f(xs, ys) * area / 4
        
    
    for i in range(len(neumann)):
        x1, y1 = coordinates[neumann[i][1]-1][1], coordinates[neumann[i][1]-1][2]
        x2, y2 = coordinates[neumann[i][2]-1][1], coordinates[neumann[i][2]-1][2]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        xs, ys = (x1 + x2) / 2, (y1 + y2) / 2
        b[neumann[i][1]-1] += g(xs, ys) * length / 2
        b[neumann[i][2]-1] += g(xs, ys) * length / 2
    
    A = assemble_stiffness_matrix(coordinates, triangles, quadrilaterals)
    u = np.zeros(len(coordinates), dtype=np.float64)
    for i in range(len(dirichlet)):
        u[dirichlet[i][1]-1] = u_d(coordinates[dirichlet[i][1]-1][1], coordinates[dirichlet[i][1]-1][2])
        u[dirichlet[i][2]-1] = u_d(coordinates[dirichlet[i][2]-1][1], coordinates[dirichlet[i][2]-1][2])
    
    b -= A @ u

    return A, b, u

def solve(coordinates, triangles, quadrilaterals, f, g, u_d, neumann, dirichlet, free_nodes):
    A, b, u = assemble_load_vector(coordinates, triangles, quadrilaterals, f, g, u_d, neumann, dirichlet)
    
    from scipy.sparse.linalg import spsolve

    # take A for only the freenodes
    
    A_free = A[free_nodes, :][:, free_nodes]
    b_free = b[free_nodes]

    print("Free nodes:", free_nodes)
    print("Reduced stiffness matrix A_free:", A_free)
    print("Reduced load vector b_free:", b_free)

    u_free = spsolve(A_free, b_free)
    print("Solution for free nodes u_free:", u_free)
    u[free_nodes] = u_free
    print("Complete solution u:", u)
    return u


def plot_solution(coordinates, triangles, quadrilaterals, u):
    import matplotlib
    matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    x = [coord[1] for coord in coordinates]
    y = [coord[2] for coord in coordinates]
    
    tri_new = []
    for tri in triangles:
        tri_new.append((tri[1]-1, tri[2]-1, tri[3]-1))
    triang = Triangulation(x, y, tri_new)
    print("Plotting solution with triangulation:", triang)
    plt.figure(figsize=(8, 6))
    for i in range(len(u)):
        if np.isnan(u[i]):
            u[i] = 0
    plt.tricontourf(triang, u, levels=14, cmap='viridis')
    plt.colorbar(label='Solution u')
    plt.title('Finite Element Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('solution.png', dpi=300)


import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.tri import Triangulation

def show(triangles, quadrilaterals, coordinates, u):
    coordinates = np.array(coordinates)[:, 1:]  # Drop first column (node index)
    x, y = coordinates[:, 0], coordinates[:, 1]
    z = np.array(u)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_xaxis()

    # Plot triangles
    if len(triangles) > 0:
        triangles_np = np.array(triangles)[:, 1:]
        triangles_np = triangles_np - 1 
        #print("Triangles :", triangles, triangles_np) # Convert to zero-based indexing
        tri = Triangulation(x, y, triangles_np)
        ax.plot_trisurf(tri, z, cmap='viridis', linewidth=0.2, antialiased=True)

    # Plot quadrilaterals (split into two triangles per quad)
    if len(quadrilaterals) > 0:
        quads_np = np.array(quadrilaterals)[:, 1:] 
        quads_np = quads_np - 1 # Drop element index
        quads_as_tris = []
        for quad in quads_np:
            quads_as_tris.append([quad[0], quad[1], quad[2]])
            quads_as_tris.append([quad[0], quad[2], quad[3]])
        tri_quad = Triangulation(x, y, np.array(quads_as_tris))
        ax.plot_trisurf(tri_quad, z, cmap='viridis', linewidth=0.2, antialiased=True)

    ax.view_init(elev=20, azim=10)
    ax.set_title("Solution of the Problem")
    plt.savefig('solution_3d.png', dpi=300)


def main():
    # Define the functions f, g, and u_d
    f = lambda x, y: 1  # Example source term
    g = lambda x, y: 0  # Example Neumann boundary condition
    u_d = lambda x, y: 0  # Example Dirichlet boundary condition

    # Solve the problem
    u = solve(coordinates, triangles, quadrilaterals, f, g, u_d, neumann, dirichlet)

    # Plot the solution
    show(triangles, quadrilaterals, coordinates, u)
    #plot_solution(coordinates, triangles, quadrilaterals, u)

# main()
