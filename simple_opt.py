import numpy as np
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor

from . import optimization

def minimize(func,bounds,args=(),xi = 0.001,max_calls = 50):
    '''
    black box function GP optimizer using built-in RBF kernel
    attempts to minimize function calls to <func>
    
    func should have the from func(x,*args)
    '''

    #use 20% of budget for random points
    dim = len(bounds)
    n_init = int(0.2*max_calls)
    n_iter = max_calls - n_init
    x_init = np.random.uniform(bounds[:,0],bounds[:,1],size=(n_init,dim)).reshape(-1,dim)
    y_init = np.array([func(ele,*args) for ele in x_init]).reshape(-1,1)

    x_sample = x_init
    y_sample = y_init
    
    #use default RBF kernel, isotropic lengthscale will be tuned automatically
    gpr = GaussianProcessRegressor(alpha=0.0001)

    for i in range(n_iter):
        gpr.fit(x_sample,y_sample)

        x_next = optimization.maximize_acquisition(\
                        optimization.expected_improvement,\
                                                   bounds,x_sample,gpr,xi)
        y_next = func(x_next,*args)

        x_sample = np.vstack((x_sample,x_next))
        y_sample = np.vstack((y_sample,y_next))

    #use trained model to get function maximum
    #(use predicted value as "acquisition function")
    #x_opt = optimization.maximize_acquisition(optimization.mean,bounds,gpr)
    #y_opt = func(x_opt,*args)

    arg = np.argmax(y_sample)
    return x_sample[arg],y_sample[arg]

def test(x):
    return -(x[0]**2 + x[1]**2) + 5

if __name__ == '__main__':
    bounds = np.array(((-2,2),(-2,2)))
    x_max,y_max = minimize(test,bounds)
    print((x_max,y_max))
    
    
    
