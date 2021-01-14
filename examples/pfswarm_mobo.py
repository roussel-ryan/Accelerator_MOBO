import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#suppress output messages below "ERROR" from tensorflow
#and prevent the use of any system GPU's
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import gpflow

from Accelerator_MOBO import mobo
from Accelerator_MOBO.optimizers import grid_search
from Accelerator_MOBO import infill
from Accelerator_MOBO.multi_objective import plotting
from Accelerator_MOBO.optimizers import evolutionary_2 

def f(x):
    f1 = np.linalg.norm(x - np.array((1,1)))
    f2 = np.linalg.norm(x + np.array((1,1)))
    return np.array((f1,f2)).reshape(-1,2)

def main():
    '''simple example of MOBO with Accelerator_MOBO
    
    In this example we wish to find the pareto front of a simple
    2 objective function (see defined above). The functions represent
    the distance between the input point and (-1,-1) or (1,1) respectively. As such 
    the pareto optimal input points lie on a line between 
    the two points (-1,-1),(1,1).

    The MOBO process requires the construction of several elements
    - Gaussian Process Regressors for each objective (using GPflow)
    - An infill (acquisition) function that defines the improvement of the 
    hypervolume
    - An optimizer function which finds the global maximum of the infill function 


    '''
    #define input domian
    bounds = np.array(((-2,2),(-2,2)))

    #sample the objective functions
    n_initial = 10
    X0 = np.random.uniform(*bounds[0],size = (n_initial,2))
    Y0 = np.vstack([f(ele) for ele in X0])

    X = X0
    Y = Y0
    
    #define objective domain
    A = np.zeros(2)
    B = np.ones(2) * 5.0

    #define kernels to be used for the gaussian process regressors
    kernels = [gpflow.kernels.RBF(lengthscales = 0.5, variance = 0.5) for i in [0,1]]

    #define GP models
    GPRs = []
    for i in range(2):
        GPRs += [gpflow.models.GPR((X,Y[:,i].reshape(-1,1)),
                                   kernels[i], noise_variance = 0.0001)]
    
    #create the optimizer object (in this case swarm search)
    swarm_opt = evolutionary_2.PFSwarmOpt(generations = 20,
                                      population = 64)

    acq = infill.UHVI(beta = 0.01)
    
    #create the mutiobjective optimizer - default infill = UHVI
    mobo_opt = mobo.MultiObjectiveBayesianOptimizer(bounds, GPRs,
                                                    B, A = A,
                                                    infill = acq)

    n_iterations = 1
    for i in range(n_iterations):
        #find next point for observation
        result = mobo_opt.get_next_point(swarm_opt.minimize)
        X_new = np.atleast_2d(result.x)
        Y_new = f(X_new)
        
        #add observations to mobo GPRs
        mobo_opt.add_observations(X_new,Y_new)


    
    #plot the hypervolume as a function of iteration #
    fig2 = plotting.plot_full(mobo_opt,f)

    
if __name__=='__main__':
    main()
    plt.show()
