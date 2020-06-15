# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:47:04 2020

@author: Ryan Roussel
"""
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

def expected_improvement(X, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(gpr.X_train_)
    
    sigma = sigma.reshape(-1, 1)
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    #logging.info(f'{mu_sample_opt} {sigma} {mu}')

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei    

def mean(X,gpr):
    ''' 
    returns mean of gpr prediction
    '''
    return gpr.predict(X)

def upper_confidence_bound(X, gpr, kappa = 0.1):
    '''
    Upper confidence bound acquisition function

    Args:
        X (TYPE): Points at which the acquisition function is evaluated at.
        gpr (TYPE): A GaussianProcessRegressor fitted to samples.
        kappa (TYPE, optional): Exploitation vs. exploration function. 
                                Defaults to 0.1.

    Returns:
        UBI at points X.

    '''
    
    mu, sigma = gpr.predict(X, return_std=True)
    f = mu + kappa*sigma
    #normalize to between 0 - 1
    #f = f - np.min(f)
    return f

def conditional_acquisition(X, X_sample, C_sample, length_scale = 1):
    '''
    Returns the conditional acquisition function

    Args:
        X (TYPE): Query Point.
        X_sample (TYPE): collection of sample locations.
        C_sample (TYPE): collection of sample feasibilities 0 or 1.

    Returns:
        Donditional value between 0 and 1 (1 == ok region).

    '''
    val = 1
    for loc,C in zip(X_sample,C_sample):
        if not C:
            val = val * norm.cdf(np.linalg.norm(X - loc),scale=length_scale)
    return val
   
def probable_improvement(X, X_sample, gpr,xi=0.0):
    #set xi to non-zero if probable improvement is too greedy
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    mu_max = np.max(mu_sample)
    
    with np.errstate(divide='warn'):
        imp = mu - mu_max - xi
        Z = imp / sigma
        ei = norm.cdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei 
    


#POSSIBLY DEPRECIATED
def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, 
                     n_restarts=25, plotting = False, C_sample=None, *args, **kwargs):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        args: args passed to acquisition function
        kwargs: kwargs passed to acquisition function

    Returns:
        Location of the acquisition function maximum.
    '''
    
    dim = X_sample.shape[1]
    min_val = 10000000
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr, *args,**kwargs)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
    
    if plotting:
        assert bounds[0].ndim == 1
        x = np.linspace(*bounds[0])
        
        fig,ax = plt.subplots()
        mu, sigma = gpr.predict(x.reshape(-1,1), return_std=True)
        ax.plot(x,mu,label='Posterior Mean')
        ax.fill_between(x,mu-sigma,mu+sigma,alpha=0.5,label='Posterior Deviation')
        ax.plot(x,-min_obj(x),label='Acquisition Func.')
        ax.plot(X_sample,Y_sample,'+')
        ax.axvline(min_x,ls='--',label='Next point')
        ax.legend()
        
    return min_x.reshape(-1, 1)

def maximize_acquisition(acquisition, bounds, *args,**kwargs):
    '''
    Find the point that maximizes the acquisition function by minimizing
    the negative of the acquisition function

    Args:
        acquisition (TYPE): Function to be maximized.
        bounds (TYPE): Function bounds.
        *args (TYPE): Positional arguments for acquisition function.
        **kwargs (TYPE): Keyword arguments for acquisition function.

    Returns:
        X value that maximizes acquisition function.

    '''
    def neg_func(X):
        return -acquisition([X],*args,**kwargs)
    
    dim = bounds.shape[0]
    min_val = 10**9
    min_x = None
    for x0 in np.random.uniform(bounds[:,0],bounds[:,1], size = (25,dim)):
        res = minimize(neg_func,x0 = x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x
    return np.array((min_x))

def main():
    pass    
    
if __name__=='__main__':
    main()
