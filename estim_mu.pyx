import numpy as np
cimport numpy as np
cimport cython

 


cdef  matrix_logarithm(np.ndarray[np.float64_t, ndim=2] A):
    cdef int n = A.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] log_A
    
    # Compute the eigenvalue decomposition of the SPD matrix A
    cdef np.ndarray[np.float64_t, ndim=2]  eigenvectors 
    cdef np.ndarray[np.float64_t, ndim=1] eigenvalues 
   
    # Take the logarithm of the eigenvalues
    cdef np.ndarray[np.float64_t, ndim=1] log_eigenvalues
    
    log_A = np.zeros((n, n), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    log_eigenvalues = np.log(eigenvalues)
    # Reconstruct the matrix logarithm using the eigenvectors and the diagonal matrix of log eigenvalues
    log_A = eigenvectors @ np.diag(log_eigenvalues) @ np.linalg.inv(eigenvectors)
    
    return log_A

cdef  matrix_sqrt(np.ndarray[np.float64_t, ndim=2] A) :
    cdef int n = A.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] sqrt_A 
    
    # Compute the eigenvalue decomposition of the SPD matrix A
    cdef np.ndarray[np.float64_t, ndim=2] eigenvectors
    cdef np.ndarray[np.float64_t, ndim=1] eigenvalues
    
    # Take the square root of the eigenvalues
    cdef np.ndarray[np.float64_t, ndim=1] sqrt_eigenvalues 
    
    sqrt_A = np.zeros((n, n), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    
    # Reconstruct the matrix square root using the eigenvectors and the diagonal matrix of sqrt eigenvalues
    sqrt_A = eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)
    
    return sqrt_A

cdef matrix_exponential(np.ndarray[np.float64_t, ndim=2] A):
    cdef int n = A.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] exp_A
    
    # Compute the eigenvalue decomposition of the SPD matrix A
    cdef np.ndarray[np.float64_t, ndim=2]  eigenvectors 
    cdef np.ndarray[np.float64_t, ndim=1] eigenvalues 
   
    # Take the exponential of the eigenvalues
    cdef np.ndarray[np.float64_t, ndim=1] exp_eigenvalues
    
    exp_A = np.zeros((n, n), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    exp_eigenvalues = np.exp(eigenvalues)
    # Reconstruct the matrix logarithm using the eigenvectors and the diagonal matrix of log eigenvalues
    exp_A = eigenvectors @ np.diag(exp_eigenvalues) @ np.linalg.inv(eigenvectors)
    
    return exp_A



cpdef  exp_cov(np.ndarray[np.float64_t, ndim=2] Y, np.ndarray[np.float64_t, ndim=2] Z):
    cdef np.ndarray[np.float64_t, ndim=2] Y_demi, Y_demi_inv, results
    Y_demi = matrix_sqrt(Y)
    Y_demi_inv = np.linalg.inv(Y_demi)
    results = np.dot(Y_demi, np.dot(matrix_exponential(np.dot(Y_demi_inv, np.dot(Z, Y_demi_inv))), Y_demi))
    return results


cpdef  log_cov(np.ndarray[np.float64_t, ndim=2] Y, np.ndarray[np.float64_t, ndim=2] Z):
    cdef np.ndarray[np.float64_t, ndim=2] Y_demi, Y_demi_inv, results
 
    Y_demi = matrix_sqrt(Y)
    Y_demi_inv = np.linalg.inv(Y_demi)
    results = np.dot(Y_demi, np.dot(matrix_logarithm(np.dot(Y_demi_inv, np.dot(Z, Y_demi_inv))), Y_demi))

    return results



cdef _log_cov(np.ndarray[np.float64_t, ndim=2] Y, np.ndarray[np.float64_t, ndim=2] Z):
    cdef np.ndarray[np.float64_t, ndim=2] Y_demi, Y_demi_inv, results
 
    
    Y_demi = matrix_sqrt(Y)
    Y_demi_inv = np.linalg.inv(Y_demi)
    results = np.dot(Y_demi, np.dot(matrix_logarithm(np.dot(Y_demi_inv, np.dot(Z, Y_demi_inv))), Y_demi))

    return results

cpdef  calcule_mu_j(np.ndarray[np.float64_t, ndim=3] Y_, np.ndarray[np.float64_t, ndim=2] Y_bar, np.ndarray[np.float64_t, ndim=1] gamma_j):
    cdef int x,y,z, i
    cdef np.ndarray[np.float64_t, ndim=2] mu
    cdef np.ndarray[np.float64_t, ndim=3] val

    (x,y,z) = np.shape(Y_)
    val = np.zeros((x,y,z), dtype=np.float64)

    for i in range(x):
        val[i,:,:] = gamma_j[i] * _log_cov(Y_bar, Y_[i,:,:])

    mu = np.sum(val,axis=0)
    

    return mu



cpdef  calcule_mu(np.ndarray[np.float64_t, ndim=3] Y, np.ndarray[np.float64_t, ndim=2] M_): 
    
    cdef int x,y,z, i
    cdef np.ndarray[np.float64_t, ndim=2] mu
    cdef np.ndarray[np.float64_t, ndim=3] val
    
    (x,y,z) = np.shape(Y)
    
    val = np.zeros((x,y,z), dtype=np.float64)
    for i in range(x):
        val[i,:,:] = _log_cov(M_,Y[i,:,:])       
    mu = np.mean(val,axis=0)
    
    return mu





