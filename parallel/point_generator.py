import numpy as np
import copy
import logging

from ..lineBO import optimizer

def generate_points(GPRs,bounds,acq,n,**kwargs):
    '''
    generate a collection of points for parallel optimization
    - the first point maximizes the acquisition function
    - subsequent points maximize the total uncertainty of GPRs
    - uses LineBO to find the maxima

    inputs:
    -------
    GPRs:      list of scikit-learn GaussianProcessRegressors
    bounds:    input space bounds
    acq:       acquisition function of the form f(x,*args)
    n:         number of points to generate

    optional inputs
    ---------------
    args:      list of arguments for acq function
    x0:        initial optimization point (default: random.uniform)
    oracle:    direction oracle for lineBO (default: random)
    verbose:   display logging messages and diagnostic plots

    '''

    logging.info('generating points for parallel computation')
    assert n > 1

    dim = len(bounds)
    x = np.empty((n,dim))
    logging.info(x)
    
    #find the point which maximizes the acquisition function
    if dim == 1:
        x[0],f = optimizer.brent_minimization(acq,bounds,kwargs.get('args',[]))
    else:                                  
        opt = optimizer.LineOpt(bounds,acq,**kwargs)
        res = opt.optimize()
        x[0] = res.x

    logging.info(f'found point that maximizes acq: {x[0]}')
    #for the rest of the points maximize the uncertaintiy acqisition function

    #make copies of each regressor
    newGPRs = [copy.deepcopy(ele) for ele in GPRs]

    logging.info('generating points to explore')
    for i in range(1,n):
        #add "new" points to each GPR training set
        #y values are any number (0.0 for simplicity) because we do not use mu
        y = np.zeros(len(x)).reshape(-1,1)
        new_x = [np.vstack((ele.X_train_,x[:i])) for ele in newGPRs]
        new_y = [np.vstack((ele.y_train_,y[:i])) for ele in newGPRs]
        logging.info(new_x)
        logging.info(new_y)
        
        #retrain GPRs
        newGPRs = [gpr.fit(x,y) for gpr,x,y in zip(newGPRs,new_x,new_y)]
        #do maximization
        if dim == 1:
            x[i],f = optimizer.brent_minimization(uncertaintity,bounds,args = newGPRs)
            
        else:
            opt2 = optimizer.LineOpt(bounds,uncertaintity,args = newGPRs)
            x[i] = opt.optimize().x

    return x

def uncertaintity(x,GPRs):
    #return the squared sum of the errors for each GPR at x
    s = 0.0
    for gpr in GPRs:
        mu,sigma = gpr.predict(np.atleast_2d(x),return_std=True)
        s = s + sigma**2
    return np.sqrt(s)
