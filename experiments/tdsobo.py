
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

class Problem:
    def __init__(self,func):
        self.func = func
        self.t = 0

    def measure(self,X):
        x = np.array((*(X.flatten()),self.t)).reshape(-1,3)
        y = np.array((self.func(X,self.t))).reshape(-1,1)
        self.t += 1
        return x,y
        
def f(x,t):
    T = 10000
    theta = 2 * np.pi * t / T
    x0 = np.array((np.cos(theta),np.sin(theta)))
    f1 = np.linalg.norm(x - x0) - 5.0

    return f1

def main():
    bounds = np.array(((-2,2),(-2,2)))
    opt(bounds)
    #plot_ground_truth(bounds)

def opt(bounds):
    '''Time dependant SOBO with GaussianProcessTools
    
    Each observation costs 1 time step


    '''
    #start the clock
    t = 0

    toy = Problem(f)
    
    #sample the objective functions over 
    n_initial = 20
    X0 = np.random.uniform(*bounds[0],size = (n_initial,2))
    X = []
    Y = []
    for i in range(n_initial):
        x, y = toy.measure(X0[i])
        X += [x]
        Y += [y]

    X = np.vstack(X)
    Y = np.vstack(Y)

    print(X)
    print(Y)

    
    #define kernels to be used for the gaussian process regressors
    kernel = gpflow.kernels.RBF(lengthscales = [1.0, 1.0, 10.0], variance = 1.5)

    #define GP models
    gpr = gpflow.models.GPR((X,Y), kernel, noise_variance = 2e-6)
    gpflow.set_trainable(gpr.likelihood.variance,False) 
    
    #create the optimizer object (in this case a simple grid search)
    #acq_opt = grid_search.GridSearch(20)
    
    acq       = infill.TDACQ(infill.UCB(beta = 0.01, maximize = False))
    
    #sobo_opt = sobo.SingleObjectiveBayesianOptimizer(bounds, gpr, acq = acq)
    sobo_opt = sobo.TDSingleObjectiveBayesianOptimizer(bounds, gpr, acq = acq)

    sobo_opt.train(1000)
    sobo_opt.print_model()
    
    time_steps = 50
    for t in range(toy.t, time_steps + toy.t):
        #find next point for observation
        sobo_opt.time = t
        result = sobo_opt.get_next_point()
        X_new = np.atleast_2d(result.x)
        Y_new = f(X_new,t).reshape(-1,1)
        t_new = np.array((t)).reshape(1,1)

        print(sobo_opt.data)

        if t % 10 == 0:
            sobo_opt.train(4000)

        #add observations to mobo GPRs
    
        sobo_opt.add_observations(X_new,Y_new,{'t':t_new})

    sobo_opt.print_model()
    sobo_opt.save('model.p')
    
def plot_ground_truth(bounds):
    n = 20
    x = np.linspace(*bounds[0],n)
    xx = np.meshgrid(x,x)
    pts = np.vstack([ele.ravel() for ele in xx]).T

    F = []
    for pt in pts:
        F += [f(pt,100)]
    F = np.array(F).reshape(-1,1)

    fig,ax = plt.subplots()
    ax.pcolor(*xx,F.reshape(n,n))
    
if __name__=='__main__':
    main()
    plt.show()
