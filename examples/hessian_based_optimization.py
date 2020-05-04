# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:15:03 2020

@author: Ryan Roussel
"""
import logging
logging.basicConfig(level=logging.INFO)
import h5py
import os

import numpy as np
import matplotlib.pyplot as plt

import numdifftools as nd

from GaussianProcessTools import gaussian_process
from GaussianProcessTools import optimization

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import prl_style
prl_style.setup()

def f(x,cov):
    return np.exp(-x.T @ cov @ x)

def physics_informed_optimization(j=None):
    bounds = np.array([[-5,5],[-5,5],[-5,5],[-5,5],[-5,5]])
    #n = 20
    dim = len(bounds)
    #x = np.linspace(*bounds[0],n)
    #y = np.linspace(*bounds[1],n)
    
    #xx,yy = np.meshgrid(x,y)
    
    #pts = np.vstack((xx.ravel(),yy.ravel())).T
    #ground_truth = np.array([f(pt) for pt in pts])
    
    correlation = 0.8
    pre = np.ones((dim,dim))*correlation + (1-correlation)*np.eye(dim)
    logging.info(pre)
    
    x0 = np.zeros((dim))
    H =  - nd.Hessian(f)(x0,pre) / (2*f(x0,pre))
    logging.info(H)
    
    n_test = 3
    X_init = np.random.uniform(*bounds[0], size = (n_test,dim))
    Y_init = [f(ele,pre) for ele in X_init]
    
    X_sample = X_init
    X_sample_f = X_init
    Y_sample = Y_init
    Y_sample_f = Y_init
   
    kernel = gaussian_process.RBFKernel(precision=H)
    gpr = gaussian_process.GPRegressor(kernel)

    m52f = RBF(length_scale=1.0)
    gprf = GaussianProcessRegressor(kernel=m52f, alpha=0.0,optimizer=None) 
    
    xi = 0.01
    
    n_iterations = 50
    for i in range(n_iterations):
        logging.info(i)
        gpr.train(X_sample,Y_sample)
        gprf.fit(X_sample_f,Y_sample_f)
    
        #EI = np.array([optimization.expected_improvement([ele], X_sample, gpr, xi) for ele in pts])
        #EIf = np.array([optimization.expected_improvement([ele], X_sample_f, gprf, xi) for ele in pts])
         
        X_next = optimization.maximize_acquisition(optimization.upper_confidence_bound,
                                                        bounds,X_sample,gpr)
        X_next_f = optimization.maximize_acquisition(optimization.upper_confidence_bound,
                                                        bounds,X_sample_f,gprf)
         
        #if i % 10 == 0:
        #    fig2,ax2 = plt.subplots(2,1)
        #    ax2[0].contourf(xx,yy,EI.reshape(n,n)/np.max(EI))
        #    ax2[1].contourf(xx,yy,EIf.reshape(n,n)/np.max(EIf))
        #    ax2[0].plot(*X_next,'r+')
        #    ax2[1].plot(*X_next_f,'r+')
           
        Y_next = f(X_next,pre)
        Y_next_f = f(X_next_f,pre)
        
        X_sample = np.concatenate((X_sample, [X_next]),axis=0)
        X_sample_f = np.concatenate((X_sample_f, [X_next_f]),axis=0)
        Y_sample = np.concatenate((Y_sample, [Y_next]),axis=0)
        Y_sample_f = np.concatenate((Y_sample_f, [Y_next_f]),axis=0)
    
    if not j == None:
        with h5py.File('hessian_trial_results.h5') as file:
            grp = file.create_group(f'{j}')
            grp.create_dataset('correlated', data=Y_sample)
            grp.create_dataset('uncorrelated',data=Y_sample_f)
        
    #fig,ax = plt.subplots(1,3)
    #ax[0].contourf(xx,yy,ground_truth.reshape(n,n),vmin=0,vmax=1)
    #ax[1].contourf(xx,yy,gpr.predict(pts).reshape(n,n),vmin=0,vmax=1)
    #ax[2].contourf(xx,yy,gprf.predict(pts).reshape(n,n),vmin=0,vmax=1)
    
    #ax[0].plot(*X_sample.T,'r+')
    #ax[0].plot(*X_sample_f.T,'g+')
        
    #fig,ax = plt.subplots()
    #max_obj = [np.max(Y_sample[:i]) for i in range(1,len(Y_sample))]
    #max_obj_f = [np.max(Y_sample_f[:i]) for i in range(1,len(Y_sample_f))]
   
    #ax.plot(max_obj)
    #ax.plot(max_obj_f)

def do_trials():
    try:
        os.remove('hessian_trial_results.h5')
    except FileNotFoundError:
        pass
    
    for j in range(50):
        physics_informed_optimization(j)

def max_obj(X):
    return [np.max(X[:i]) for i in range(1,len(X))]

def plot_trials():
    n_trials = 50
    cons = []
    uncons = []
    with h5py.File('hessian_trial_results.h5') as file:
        for i in range(50):
            cons.append(max_obj(file[f'{i}']['correlated'][:]))
            uncons.append(max_obj(file[f'{i}']['uncorrelated'][:]))
            
    cons = np.array(cons)
    uncons = np.array(uncons)
    
    fig,ax = plt.subplots(2,1,sharex=True)
    for i in range(n_trials):
        ax[1].plot(cons[i],'C1',alpha=0.1)
        ax[0].plot(uncons[i],'C0',alpha=0.1)
        
    ax[0].plot(np.mean(uncons,axis=0),'C0',label='Un-correlated kernel')
    ax[1].plot(np.mean(cons,axis=0),'C1',label='Correlated kernel')
    
    ax[0].set_ylabel('Objective')
    ax[1].set_ylabel('Objective')
    ax[1].set_xlabel('Step')
    for ele in ax:
        ele.legend()
    
    fig.savefig('hessian_5d_trials.svg')

def main():
    #do_trials() 
    plot_trials()
    #physics_informed_optimization()
    
if __name__=='__main__':
    main()
