import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#suppress output messages below "ERROR" from tensorflow
#and prevent the use of any system GPU's
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import gpflow

from GaussianProcessTools import sobo
from GaussianProcessTools.optimizers import grid_search
from GaussianProcessTools import infill
from GaussianProcessTools.multi_objective import plotting

def f(x):
    f1 = np.linalg.norm(x - np.array((1,1)))
    return f1.reshape(-1,1)

def main():
    '''simple example of SOBO with GaussianProcessTools
    
    The MOBO process requires the construction of several elements
    - Gaussian Process Regressors for each objective (using GPflow)
    - An infill (acquisition) function that defines the improvement of the 
    hypervolume
    - An optimizer function which finds the global maximum of the infill function 


    '''
    #define input domian
    bounds = np.array(((-2,2),(-2,2)))

    #sample the objective functions
    n_initial = 5
    X0 = np.random.uniform(*bounds[0],size = (n_initial,2))
    Y0 = np.vstack([f(ele) for ele in X0])

    X = X0
    Y = Y0
    
    #define kernels to be used for the gaussian process regressors
    kernel = gpflow.kernels.RBF(lengthscales = 1.0, variance = 0.5)

    #define GP models
    gpr = gpflow.models.GPR((X,Y), kernel, noise_variance = 0.0001)

    print(gpr.predict_y(np.zeros(2).reshape(-1,2)))
    
    #create the optimizer object (in this case a simple grid search)
    #acq_opt = grid_search.GridSearch(20)
    acq = infill.UCB(beta = 2.0, maximize=False)

    
    sobo_opt = sobo.SingleObjectiveBayesianOptimizer(bounds, gpr, acq = acq)

    n_iterations = 30
    for i in range(n_iterations):
        #find next point for observation
        result = sobo_opt.get_next_point()
        X_new = np.atleast_2d(result.x)
        Y_new = f(X_new)

        print(sobo_opt.data)
        
        #add observations to mobo GPRs
        sobo_opt.add_observations(X_new,Y_new)

    
if __name__=='__main__':
    main()
    plt.show()
