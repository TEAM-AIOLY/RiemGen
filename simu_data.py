import numpy as np
import scipy as sp
from scipy.linalg import expm,logm,sqrtm
from scipy.stats import norm 
from distinctipy import distinctipy
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import time

import covGen as cg



start = time.time()

K=6
p=3
Rho=[-0.7,-0.01,0.5,0.01,0.99,-0.9]
Sigma=[0.5,0.3,0.1,0.3,0.7,0.25]
Y= cg.Gen_cov_mixture(K=K,p=p,Sigma=Sigma,Rho=Rho,Nmix=[3000,2000,1500,2500,1000,1000])

end=time.time()
print("generation duration: = " + str(end-start) +" seconds")

start = time.time()
Nmix = [len(y) for y in Y]
N = np.sum(Nmix)
logdet_Y = []
for i in range(len(Nmix)):
    logdet_Y_i = np.zeros((1, Nmix[i]))
    logdet_Y_i_2 = np.zeros((1, Nmix[i]))
    for j in range(Nmix[i]):
        logdet_Y_i[0, j] = np.trace(sp.linalg.logm(Y[i][j]))       
    logdet_Y.append(logdet_Y_i)

end=time.time()
print("logdet computaion duration: = " + str(end-start) +" seconds")    

logdet_Y =[np.sort(ly) for ly in logdet_Y]
mean_ly =[np.mean(ly) for ly in logdet_Y]
sigma_ly =[np.var(ly) for ly in logdet_Y]

x = [np.linspace(np.min(ly), np.max(ly), 50) for ly in logdet_Y]
y = [ norm.pdf(x[i],mean_ly[i], np.sqrt(sigma_ly[i])) for i in range(len(logdet_Y)) ]



# color_list = distinctipy.get_colors(len(logdet_Y)+3,pastel_factor=0.4)

color_list = ['blue', 'orange', 'green', 'red','purple','brown','yellow','yellow']

labels = ['G_1', 'G_2', 'G_3']
fig, ax = plt.subplots()
for i in range(len(logdet_Y)):
    ax.hist(logdet_Y[i][0,:], bins=25, color=color_list[i], alpha=0.5,label=f'G_{i+1}',density='density')
    ax.plot(x[i], y[i], linewidth=3, color=color_list[i],alpha=1)
   
    ax.legend()
plt.show()

logdet_mix =np.concatenate(logdet_Y,axis=1)

x = np.linspace(np.min(logdet_mix), np.max(logdet_mix), len(Nmix)*50) 

gmm=[]
for i in range (len(Nmix)):
    y = (Nmix[i]/N)*norm.pdf(x,mean_ly[i], np.sqrt(sigma_ly[i]))
    gmm.append(y)
    
gmm =np.sum(gmm, axis=0)

fig,ax=plt.subplots()
ax.hist(logdet_mix[0,:],bins=300, color=color_list[len(Nmix)+1], alpha=0.5,label='Gmm',density=True)
ax.plot(x, gmm, linewidth=3, color='black',alpha=1)
ax.legend()
plt.show()
