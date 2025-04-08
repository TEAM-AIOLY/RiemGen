import numpy as np
import scipy as sp
from scipy.stats import uniform
from scipy.linalg import expm,logm,sqrtm,qr

"""
Zexing Yao, Florent Abdelghafour
# Generating a Gaussian mixture distribution of covariance matrices

K = number of modes
p = dimension of covariance matices
Nmix = number of samples in each mode
Sigma = [s_1,..., S_K], the dispersions of each mode
Rho = [r_1,..., r_k], parameters to generate [M_1,....,M_k]
                      the centres of Mass of each mode

"""
def Gen_cov_mixture(K=3, p=3, Nmix=None, Sigma=None, Rho=None):
    
    if Nmix is None:
        Nmix = [2000] * K
    elif len(Nmix) != K:
        raise ValueError("Length of Nmix must be equal to K")
        
    if Sigma is None:
        Sigma = [1.0] * K
    elif len(Sigma) != K:
        raise ValueError("Length of sigma must be equal to K")
        
    if Rho is None:
        Rho = [0.5] * K
    elif len(Rho) != K:
        raise ValueError("Length of rho must be equal to K")
        
    ########### Generate centres of mass from factor Rho   ###########
############### in practice we need the square root ##################     
    M = [cov_sigma(p, rho) for rho in Rho]   
    Msqrt = [sqrtm(m) for m in M]
#######################################################################

   
    Ri =[generate_ri(sigma,p,N) for sigma,N in zip(Sigma,Nmix) ]
   

    Y = [np.zeros((N,p,p)) for N in Nmix]
    Z =[np.random.rand(N,p,p) for N in Nmix]
    
    D = [np.zeros((N,p,p)) for N in Nmix]
    Q =[np.zeros((N,p,p)) for N in Nmix]    
     # R =[np.zeros((N,p,p)) for N in Nmix]  
        
    for i,(z,ri) in enumerate(zip(Z,Ri)):
        for j in range(z.shape[0]):
           
            q,r = qr(z[j,:,:])                                 
            d = np.diag(np.exp(ri[j,:]))                
            y =np.dot(q.T, np.dot(d, q))
            y_centred = np.dot(np.dot(np.transpose(Msqrt[i]), y), Msqrt[i])
                               
            Y[i][j,:,:]= y_centred    
        
    return Y


def cov_sigma(p, rho):
    if rho < -1 or rho > 1:
        print("Warning: rho must be within the range [-1, 1]. Setting rho to 1.")
        rho = 1   
    sigma = np.eye(p, dtype='float')
    
    for i in range(p):
        for j in range(p):
            sigma[i][j] = rho**(abs(i-j))
    
    return sigma

# def qr(A):
#     """QR decomposition par La méthode de Gram-schmidt"""
#     Q=np.zeros_like(A)
#     cnt = 0
#     for a in A.T:
#         u = np.copy(a)
#         for i in range(0, cnt):
#             u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i]) 
#         e = u / np.linalg.norm(u)  
#         Q[:, cnt] = e
#         cnt += 1
#     R = np.dot(Q.T, A)
#     return (Q, R)



def unifpdf_dim_p(x, a, b):
    n = x.shape[0]
    val = np.zeros((n, 1))
    for i in range(n):
        pdf_values = uniform.pdf(x[i, :], loc=a, scale=b-a)
        val[i, 0] = np.prod(pdf_values)
    return val


def prod_sinh(x, d):
    p = 1
    for i in range(d-1):
        for j in range(i+1, d):
            p *= np.sinh(abs(x[i]-x[j])/2)
    return p


def pdf(x, gamma=1.0, d=2):
    return np.exp(-np.sum(x**2) / (2 * gamma**2)) * prod_sinh(x, d)

def proppdf(x, y, delta, d):
    return np.prod(unifpdf_dim_p(np.array([y - x]), -delta, delta))

def proprnd(x, delta):
    return x + np.random.rand(x.shape[0]) * 2 * delta - delta


def mhsample(N, pdf, proprnd, proppdf, gamma, delta, d,burnin=0, thin=1):
    
    x0 = np.random.rand(d)
    chain = np.zeros((N+burnin, d))
    chain[0,:] = x0
    acceptance_rate = 0.0
    
    for i in range(1, (N+burnin)*thin):
        x = chain[(i-1)//thin,:]
        y = proprnd(x, delta)
        
        alpha = min(1, pdf(y, gamma, d) * proppdf(y, x, delta, d) / (pdf(x, gamma, d) * proppdf(x, y, delta, d)))
        u = np.random.rand()
         
        if u < alpha:
            chain[i//thin,:] = y
            acceptance_rate += 1 / (N+burnin)
        else:
           chain[i//thin,:] = x          

    return chain[burnin:][::thin]


def generate_ri(gamma=1.0,d=2,N=10000):
    
    burnin=1000 # number of discarded samples
    thin=1
    delta = gamma
    
    r= mhsample(N, pdf, proprnd, proppdf, gamma, delta, d,burnin,thin)    

    return(r)








