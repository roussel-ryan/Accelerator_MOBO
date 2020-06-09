import numpy as np
import time

from scipy.optimize import minimize

from . import EI_tools as EIT
from . import pareto_tools as PT

from ..lineBO import optimizer as lineOpt
from ..lineBO import oracles


import logging

def get_next_point(F,GPRs,bounds,r,x0,**kwargs):
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
    A = kwargs.get('A',np.zeros((2)))
    B = kwargs.get('B',r)
    
    if dim == 2:
    
        #new_point = layered_minimization(get_EHVI, bounds, args = (GPRs,F,r,A,B))
        s = PT.get_PF(F)
        s = PT.sort_along_first_axis(s)
        fargs = [GPRs,s,r,A,B]
        obj = EIT.get_EHVI
        lineBO = lineOpt.LineOpt(bounds,oracles.random,obj,args = fargs,x0=x0)
        res = lineBO.optimize()

        #if distance between new point and old point is small, 
        #start LineOpt at a random location
        norm_dist = np.linalg.norm(res.x - x0) / np.linalg.norm(x0)
        logging.info(f'normalized distance between opt.x and x0: {norm_dist}')
        logging.info(f'res.x: {res.x}')
        logging.info(f'x0: {x0}')
        if norm_dist < 1e-3:
            x0 = np.random.uniform(bounds[:,0],bounds[:,1])
            lineBO = lineOpt.LineOpt(bounds,oracles.random,obj,args = fargs,x0=x0)
            res = lineBO.optimize()
            logging.info(f'new optimized point: {res.x}')

        return res.x
        
    elif dim == 3:
        pass
    else:
        print('can\'t do higher dimentional problems yet!')


        
def layered_minimization(func,bounds,n_restarts = 25, args=()):
    min_val = 10**20
    dim = len(bounds)
    nfev = 0

    s = time.time()
    for x0 in np.random.uniform(bounds[:,0],bounds[:,1], size = (n_restarts,dim)):
        res = minimize(func, x0 = x0, args = args,bounds = bounds, method='L-BFGS-B')#,tol = 0.001)
        nfev = nfev + res.nfev
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
    #logging.info(f'number of function evaluations {nfev}, avg exec time {(time.time() - s)/nfev}')
    return min_x
