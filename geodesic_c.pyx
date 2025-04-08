import numpy as np
import scipy as sp
cimport numpy as np

cpdef d_rao_fisher(np.ndarray[np.float64_t, ndim=3] C1, np.ndarray[np.float64_t, ndim=2] C2, method=None):
    cdef int N, p, q
    cdef np.ndarray[np.float64_t, ndim=1] lambda_vals
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1] distances
    cdef np.ndarray[np.float64_t, ndim=2] c

    N, p, q = C1.shape[0], C1.shape[1], C1.shape[2]

    distances = np.zeros(N, dtype=np.float64)  # Array to store the distances
    for i in range(N):
        c = np.asarray(C1[i, :, :], dtype=np.float64)
        lambda_vals = sp.linalg.eigvalsh(C2, c)

        lambda_vals = np.log(lambda_vals)
        d = np.sqrt(np.sum(lambda_vals**2))
        distances[i] = d

    return distances