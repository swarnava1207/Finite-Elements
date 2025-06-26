from math import *
import scipy.sparse as sp
import numpy as np
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
 # Use a non-interactive backend for saving plots
def derivative(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)
def lagrange_pol(f,x,x_list,n):
    pol = 0
    for i in range(n+1) :
        #print("pol:", x_list[i],i,f(x_list[i]))
        pol += Lag(n,i,x,x_list)*f(x_list[i])
    return pol
def Lag(n,k,x,x_list):
    prod = 1
    for i in range(n+1):
        if i == k:
            continue
        prod *= (x - x_list[i])/(x_list[k] - x_list[i])
    return prod

def equality(x,y):
    if abs(x - y) < 1e-8:
        return True
    return False
def zero_one(x,x_list,i):
    #print(x,x_list,i)
    for j in range(len(x_list)):
        #print("Checking:", j, x, x_list[j], i)
        if equality(x,x_list[j]) and j == i:
            return 1
        elif equality(x,x_list[j]) and j != i:
            return 0
        #print("all fine")
def Bases(a,b,n,r,x) :
    end_list = []
    other_list = []
    partition = [(a + i*(b-a)/n) for i in range(n+1)]
    partition_table = [[partition[i] + j*(partition[i+1] - partition[i])/r for j in range(r+1)] for i in range(n)]
    def end(x,i) :
        if partition[i-1] <= x <= partition[i]:
            return lagrange_pol(lambda x : zero_one(x,partition_table[i-1],r),x,partition_table[i-1],r)
        elif partition[i] < x <= partition[i+1]:
            return lagrange_pol(lambda x : zero_one(x,partition_table[i],0),x,partition_table[i],r)
        else :
            return 0
    def other(x,i,j):
        if partition[i] <= x <= partition[i+1]:
            return lagrange_pol(lambda x : zero_one(x,partition_table[i],j),x,partition_table[i],r)
        else :
            return 0
    for i in range(1,n):
        end_list.append(end(x,i))
    for i in range(n):
        for j in range(1,r):
            other_list.append(other(x,i,j))
    return end_list + other_list

def Gauss_Seidel(A,b,x_0,n,TOL):
    sum1 = sum2 = 0
    x = [0]*n
    c = 0
    while True:
        
        #x = [0]*n
        for i in range(0,n):
            sum1 = sum2 = 0
            for j in range(0,i):
                sum1 = sum1 + A[i][j]*x[j]
            for j in range(i+1,n):
                sum2 = sum2 + A[i][j]*x_0[j]
            x[i] = (b[i] - sum1 - sum2)/A[i][i]
        # if c % 100 == 0:
        #     print("Iteration", c + 1, ":", x)
        #print("Iteration",c+1,":",x)
        #time.sleep(0.01)
        # print("Checking the convergence criteria")
        # print(max([abs(x[i]-x_0[i]) for i in range(0,n)]),max([abs(x[i]) for i in range(n)]))
        if max([abs(x[i]-x_0[i]) for i in range(0,n)])/max([abs(x[i]) for i in range(n)]) < TOL:
            # print("Checking the convergence criteria")
            # print(max([abs(x[i]-x_0[i]) for i in range(0,n)]),max([abs(x[i]) for i in range(n)]))
            # print(max([abs(x[i]-x_0[i]) for i in range(0,n)])/max([abs(x[i]) for i in range(n)]))
            return x
        
        x_0 = x.copy()
            
        c = c + 1

import numpy as np
import scipy.sparse as sp

def gauss_seidel_sparse(A_sparse, b, x0, TOL, max_iter=10000, verbose=False):
    """
    Optimized Gauss-Seidel using sparse CSR matrix format.

    Parameters:
    - A_sparse: scipy.sparse.csr_matrix
    - b: numpy array
    - x0: initial guess (numpy array)
    - TOL: tolerance for convergence
    - max_iter: maximum number of iterations
    - verbose: print iterations if True

    Returns:
    - x: solution vector
    """
    A = A_sparse.tocsr()
    n = A.shape[0]
    x = np.copy(x0)
    
    for c in range(max_iter):
        x_old = np.copy(x)
        
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            sum_val = 0.0
            diag = 0.0
            
            for idx in range(row_start, row_end):
                j = A.indices[idx]
                A_ij = A.data[idx]
                
                if j == i:
                    diag = A_ij
                else:
                    sum_val += A_ij * x[j]  # Note: x[j] can be from old or new depending on j<i or j>i
                
            x[i] = (b[i] - sum_val) / diag

        # Convergence check
        diff_norm = np.linalg.norm(x - x_old, ord=np.inf)
        rel_norm = np.linalg.norm(x, ord=np.inf)
        
        if verbose and c % 100 == 0:
            print(f"Iteration {c + 1}: {x}")
        
        if diff_norm / rel_norm < TOL:
            return x

    raise RuntimeError("Gauss-Seidel did not converge within the maximum number of iterations.")





def trapezoidal(a,b,f):
    return (b-a)*(f(a)+f(b))/2
def simpsons(a,b,f):
    return (b-a)*(f(a)+f(b)+4*f((a+b)/2))/6

def comp_simpsons(a,b,n,f):
    x_val = []
    for i in range(n+1):
        x_val.append(a+i*(b-a)/n)
    integral = 0
    for i in range(n):
        integral += simpsons(x_val[i],x_val[i+1],f)
    return integral

def inner_product(f,g,k,a,b):
    func = lambda x: f(x)*g(x)* k(x) + derivative(f,x,1e-3)*derivative(g,x,1e-3)
    #print("Checking for 0",func(0.5),func(0.75))
    return comp_simpsons(a, b, 100, func)
def L(f,g,a,b):
    return comp_simpsons(a, b, 100, lambda x: f(x) * g(x))

# def pivot_if_needed(A_sparse, b_vector):
#     A = A_sparse.toarray()
#     n = len(A)
#     for i in range(n):
#         if abs(A[i][i]) < 1e-10:
#             # Search for a row below with a non-zero at column i
#             for j in range(i + 1, n):
#                 if abs(A[j][i]) > 1e-10:
#                     print("found")
#                     # Swap rows i and j in A and b
#                     A[[i, j]] = A[[j, i]]
#                     b_vector[i], b_vector[j] = b_vector[j], b_vector[i]
#                     break
#     return sp.csr_matrix(A), b_vector

def FEM(a,b,n,r,f,k):
    #print("size:",len(Bases(a,b,n,r,x)))
    dimension = r*n - 1
    A = [[0 for _ in range(dimension)] for _ in range(dimension)]
    b_vector = [0 for _ in range(dimension)]
    x_list = [a + i*(b-a)/n for i in range(n+1)]
    for i in range(dimension):
        for j in range(dimension):
            A[i][j] = inner_product(lambda x: Bases(a,b,n,r,x)[i], lambda x: Bases(a,b,n,r,x)[j], k, a, b)
    for i in range(dimension):
        b_vector[i] = L(lambda x: Bases(a,b,n,r,x)[i], f, a, b)
    # A_sparse = sp.csr_matrix(A)
    # A_sparse, b_vector = pivot_if_needed(A_sparse, b_vector)
    # A = A_sparse.toarray()
    x_0 = [1 for _ in range(dimension)]
    TOL = 1e-6
    A = np.array(A)
    b_vector = np.array(b_vector)
    alphas = Gauss_Seidel(A, b_vector, x_0, dimension, TOL)
    result = lambda x: sum(alphas[i] * Bases(a, b, n, r, x)[i] for i in range(dimension))
    return result

def plot(f,k,a,b,n,r,sol,fig):
    
    x_values = np.linspace(a, b, 50)
    y_values = [sol(x) for x in x_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Original', color='blue')
    
    func = FEM(a, b, n, r, f, k)
    y_sol_values = [func(x_values[i]) for i in range(len(x_values))]
    plt.plot(x_values, y_sol_values, label='FEM Solution', color='red', linestyle='--')
    plt.legend()
    plt.savefig(f'fem_solution_{fig}.png')

def f1(x):
    return pi**2 * sin(pi * x)
def sol1(x):
    return sin(pi * x)

#plot(f1, lambda x: 0, 0, 1, 20, 2,sol1,1)

def f2(x):
    return pi**2 * sin(pi * x) + sin(pi * x)
def sol2(x):
    return sin(pi * x)

#plot(f2, lambda x : 1, 0, 1, 4, 1, sol2,2)

def sparse_FEM(a,b,n,r,f,k):
    dimension = r*n - 1
    A = sp.csr_matrix((dimension, dimension))
    b_vector = np.zeros(dimension)
    x_list = [a + i*(b-a)/n for i in range(n+1)]
    
    for i in range(dimension):
        for j in range(dimension):
            A[i, j] = inner_product(lambda x: Bases(a, b, n, r, x)[i], lambda x: Bases(a, b, n, r, x)[j], k, a, b)
    
    for i in range(dimension):
        b_vector[i] = L(lambda x: Bases(a, b, n, r, x)[i], f, a, b)
        
    
    x_0 = np.ones(dimension)
    TOL = 1e-6
    alphas = gauss_seidel_sparse(A, b_vector, x_0, TOL)
    
    result = lambda x: sum(alphas[i] * Bases(a, b, n, r, x)[i] for i in range(dimension))
    return result


            
def main():
    a, b = 0, 1
    n = 20
    r = 3
    f = f1
    k = lambda x: 0
    sol = sol1
    start_normal = time.time()
    result = FEM(a, b, n, r, f, k)
    end_normal = time.time()
    start_sparse = time.time()
    result_sparse = sparse_FEM(a, b, n, r, f, k)
    end_sparse = time.time()
    
    print("\n\n\n")
    print("Time taken by normal FEM :", end_normal - start_normal)
    print("Time taken by sparse FEM :", end_sparse - start_sparse)

main()