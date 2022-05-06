import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


Traj_F = np.array([[1,2],[3,4]])
#Traj_F = np.array([1,2,3,4])
Traj_L = np.array([[5,6],[7,8]])
#Traj_L = np.array([5,6,7,8])
Traj = np.array([[1,0],[3,4]])
#Traj = np.array([1,0,3,4])

def mvnpdf(traj,traj_F,traj_L):
    traj = traj.flatten()
    traj_F = traj_F.flatten()
    traj_L = traj_L.flatten()
    num_states = traj.shape[0]
    #print(num_states)
    
    MEAN = np.zeros(num_states)
    COV = 0.1*np.eye(num_states)
    pdf_F = multivariate_normal.pdf(traj-traj_F, mean=MEAN, cov=COV)
    pdf_L = multivariate_normal.pdf(traj-traj_L, mean=MEAN, cov=COV)
    #print(pdf_F, pdf_L)
    
    return pdf_F, pdf_L

#mvnpdf(Traj, Traj_F, Traj_L)

