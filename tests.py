def sparse_matrix(A):
    from scipy.sparse import csr_matrix
    return csr_matrix(A)

A = [[1, 0, 0],[0, 0, 3],[4, 0, 0]]
if __name__ == "__main__":
    print(sparse_matrix(A))
    print(sparse_matrix(A).todense())