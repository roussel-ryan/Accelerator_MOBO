import numpy as np
import time

from scipy.optimize import minimize

from . import EI_tools as EIT
from . import pareto_tools as PT

import logging

def get_next_point(F,GPRs,bounds,r):
    '''
    get the next evaluation point for X based on expected hypervolume improvement
    -----------------------------------------------------------

    X: array of input space vectors
    F: array of objective space vectors
    GPRs: list of scikit-learn GaussianProcessRegressors that have been trained 
             on (X,F)
    bounds: array specifying input space boundaries
    r: reference point

    -----------------------------------------------------------
    output: point in input parameter space that maximizes EHVI
    '''
    dim = F.shape[1]

    assert dim == len(GPRs),'# of gaussian processes != objective space!'
    
    if dim == 2:
    
        new_point = layered_minimization(get_EHVI, bounds, args = (GPRs,F,r))
        return new_point
        
    elif dim == 3:
        pass
    else:
        print('can\'t do higher dimentional problems yet!')

def get_EHVI(X,GPRs,F,r):
    '''
    x: point input from optimizer
    F: set of observed points
    GPRs: list of GP regressors
    r: reference point
    '''
    #logging.info(f'calling get_EHVI() with point:{x.reshape(-1,2)}')
    S = PT.get_non_dominated_set(F)
    S = PT.sort_along_first_axis(S)[::-1]

    dim = len(X)
    f = np.array([ele.predict(X.reshape(-1,dim),return_std=True) for ele in GPRs]).T[0]
    #logging.info(f)
    ehvi = -EIT.EHVI_2D(f[0],f[1],S,r)
    #logging.info((f[0],f[1],ehvi))
    return ehvi

        
def layered_minimization(func,bounds,n_restarts = 10, args=()):
    min_val = 10000000
    dim = len(bounds)
    nfev = 0

    s = time.time()
    for x0 in np.random.uniform(bounds[:,0],bounds[:,1], size = (n_restarts,dim)):
        res = minimize(func, x0 = x0, args = args,bounds = bounds, method='L-BFGS-B',tol = 0.001)
        nfev = nfev + res.nfev
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
    #logging.info(f'number of function evaluations {nfev}, avg exec time {(time.time() - s)/nfev}')
    return min_x
