from math import *
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np

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
        if (c+1) % 100 == 0:
            #print("Iteration",c+1,":",x)
            pass
        #time.sleep(0.01)
        # print("Checking the convergence criteria")
        # print(max([abs(x[i]-x_0[i]) for i in range(0,n)]),max([abs(x[i]) for i in range(n)]))
        if max([abs(x[i]-x_0[i]) for i in range(0,n)])/max([abs(x[i]) for i in range(n)]) < TOL:
            # print("Checking the convergence criteria")
            # print(max([abs(x[i]-x_0[i]) for i in range(0,n)]),max([abs(x[i]) for i in range(n)]))
            # print(max([abs(x[i]-x_0[i]) for i in range(0,n)])/max([abs(x[i]) for i in range(n)]))
            print("Convergence achieved after", c+1, "iterations")
            return x
        
        x_0 = x.copy()
        c = c + 1

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

def phi(x,x_list,i):
    if x_list[i-1] <= x <= x_list[i] :
        return (x - x_list[i-1])/(x_list[i] - x_list[i-1])
    elif x_list[i] < x <= x_list[i+1] :
        return (x_list[i+1] - x)/(x_list[i+1] - x_list[i])
    else :
        return 0
    
def der_phi(x,x_list,i):
    if x_list[i-1] <= x <= x_list[i] :
        return 1/(x_list[i] - x_list[i-1])
    elif x_list[i] < x <= x_list[i+1] :
        return -1/(x_list[i+1] - x_list[i])
    else :
        return 0

def inner_product(a,b,f,g,der_f,der_g,k):
    func = lambda x : f(x)*g(x)*k(x) + der_f(x)*der_g(x)
    return comp_simpsons(a, b,1000,func)

def L(f,g,a,b):
    func = lambda x : f(x)*g(x)
    return comp_simpsons(a, b,1000, func)

def FEM_first_order(a, b, n, f, k):
    partition = [(a + i*(b-a)/n) for i in range(n+1)]
    
    A = [[0 for _ in range(n-1)] for _ in range(n-1)]
    b_vector = [0 for _ in range(n-1)]
    
    for i in range(n-1):
        for j in range(n-1):
                A[i][j] = inner_product(a, b, 
                                        lambda x: phi(x, partition, i+1), 
                                        lambda x: phi(x, partition, j+1), 
                                        lambda x: der_phi(x, partition, i+1),
                                        lambda x: der_phi(x, partition, j+1),
                                        k)
        b_vector[i] = L(lambda x: phi(x, partition, i+1), f, a, b)
    x_0 = [0] * (n-1)
    TOL = 1e-20
    alphas = Gauss_Seidel(A, b_vector, x_0, n-1, TOL)
    result = lambda x : sum(alphas[i] * phi(x, partition, i+1) for i in range(n-1))
    return result

def plot(f,k,a,b,n,sol):
   
    x_values = np.linspace(a, b, 500)
    y_values = [sol(x) for x in x_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Original', color='blue')
    
    function = FEM_first_order(a, b, n, f, k)
    y_sol_values = [function(x) for x in x_values]
    plt.plot(x_values, y_sol_values, label='FEM Solution', color='red', linestyle='--')
    plt.savefig('fem_solution.png')

def f1(x):
    return pi**2 * sin(pi * x)
def sol1(x):
    return sin(pi * x)

plot(f1, lambda x: 0, 0, 1, 10,sol1)
                