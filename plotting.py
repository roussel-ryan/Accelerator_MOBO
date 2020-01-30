# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:05:06 2020

@author: Ryan Roussel
"""
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def plot_1D_GP(X,X_sample,Y_sample,gpr):
    pass

def plot_2D_acquisition(acquisition, X_sample, Y_sample, gpr, bounds, *args, **kwargs):
    dim = X_sample.shape[1]
    n_restarts = 25
    min_val = 10000000
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr, *args,**kwargs)
    
    # Find the best optimum by starting from n_restart different random points.
    #for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
    #    res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
    #    if res.fun < min_val:
    #        min_val = res.fun[0]
    #        min_x = res.x
            
    fig,ax = plt.subplots()
    n = 20
    x = np.linspace(*bounds[0],n)
    y = np.linspace(*bounds[1],n)
    xx,yy = np.meshgrid(x,y)
    acq = np.zeros_like(xx).ravel()
    for i in range(len(xx.ravel())):
        acq[i] = -min_obj(np.vstack((xx.ravel()[i],yy.ravel()[i])))
    
    im = ax.contourf(xx,yy,acq.reshape(n,n))
    fig.colorbar(im)
    logging.info(X_sample)
    for ele in X_sample:
        ax.plot(*ele,'+r')

def get_func_value(X,func,*args,**kwargs):
    vals = np.zeros(len(X))
    for i in range(len(X)):
        vals[i] = func(X[i],*args,**kwargs)
 
    return vals    
    

def main():
    pass
    
if __name__=='__main__':
    main()
