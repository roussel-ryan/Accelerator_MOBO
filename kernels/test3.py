import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary, set_trainable
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
import numdifftools as nd

import hessian_RBF

dim = 2

def f(X,cov):
    #return multivariate_normal.pdf(X,mean=np.zeros(dim),cov=cov)
    return multivariate_normal.pdf(X,mean=np.zeros(dim),cov=cov)/multivariate_normal.pdf(np.zeros(dim),mean=np.zeros(dim),cov=cov)



def main():
    #specify bounds
    bounds = np.array((-5,5))
    
    #specify cov
    diag = True
    if diag:
        add = np.zeros(dim)
        add[::2] = 0.5
        d_ele = np.ones(dim) + np.arange(dim)*0.1 
        cov = np.diag(d_ele**2)

    else:
        cov = np.diag((np.ones(dim) + 0.1*np.arange(dim))**2) / 2
        cov += np.diag(0.1*np.arange(dim-1),k=1)
        cov += cov.T
        d_ele = np.ones(dim)

    S = np.ones(dim)

    kernel = hessian_RBF.HessianRBF()

    x0 = np.zeros(dim)
    
    kernel.update_precision(x0,f,cov)
    print(kernel.S)

if __name__=='__main__':
    main()
    plt.show()
