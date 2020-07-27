import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary, set_trainable
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal
import numdifftools as nd

import correlated

def f(X,cov):
    return multivariate_normal.pdf(X,mean=np.zeros(2),cov=cov)

def main():
    #specify bounds
    bounds = np.array(((-3,3),(-3,3)))

    #specify cov
    cov = np.diag(np.array((1,2))**2)
    print(cov)
    
    #create training points
    n_train = 25

    x = np.linspace(*bounds[0],n_train)
    pts = np.meshgrid(x,x)
    X = np.vstack((pts[0].ravel(), pts[1].ravel())).T
    Y = f(X,cov).reshape(-1,1)
    #    print(X)
    #print(Y)

    base_kernel = gpflow.kernels.RBF(variance=1e-5,lengthscales=[1.5,3.])
    
    HP = {'L':np.array((1.0,0.5))}
    test_kernel = correlated.AdvancedRBF(hyper_parameters = HP,
                                         mode = 'anisotropic')

    #chain_elements = [tfp.bijectors.Exp(),tfp.bijectors.Softplus(0.001)]
    #chain_elements = [tfp.bijectors.Exp()]
    chain_elements = [tfp.bijectors.Softplus(0.0001)]
    
    transform = tfp.bijectors.Chain(bijectors = chain_elements)
    #print(transform.forward(-np.inf))
    
    #test_kernel.L.transform = transform
    base_kernel.lengthscales.transform = transform

    
    model = gpflow.models.GPR(data = (X,Y),
                              kernel = base_kernel)

    
    
    model.likelihood.variance.assign(1e-5)
    model.kernel.variance.assign(1e-5)
    
    set_trainable(model.likelihood.variance,False)
    set_trainable(model.kernel.variance,False)
    
    lengthscale_bounds = (0.01, 10)
    n_samples = 2
    l = np.linspace(*lengthscale_bounds,n_samples)
    lpts = np.meshgrid(l,l)
    L = np.vstack((lpts[0].ravel(),lpts[1].ravel())).T

    lml = []
    uc_var = []
    for pt in L:
        #model.kernel.L.assign(pt)
        model.kernel.lengthscales.assign(pt)
        uc_var.append(model.kernel.lengthscales.unconstrained_variable.numpy())
        lml.append(model.log_marginal_likelihood())

    uc_var = np.array(uc_var).T
    lml = np.array(lml)

    fig,ax = plt.subplots()
    #c = ax.contourf(lpts[0],lpts[1],lml.reshape(n_samples,n_samples),levels=100)
    c = ax.contourf(uc_var[0].reshape(n_samples,n_samples),
                uc_var[1].reshape(n_samples,n_samples),
                lml.reshape(n_samples,n_samples),levels=100)
                
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    fig.colorbar(c,ax=ax)

    #do optimization
    model.kernel.lengthscales.assign((3.0,3.0))
    print(model.kernel.lengthscales.unconstrained_variable)
    #model.kernel.L.assign((5.0,5.0))
    #opt = gpflow.optimizers.Scipy()
    opt = tf.optimizers.SGD(learning_rate=0.0001)

    for i in range(1000):
        #print(model.kernel.L.numpy())
        opt.minimize(model.training_loss,model.trainable_variables)

    
    print_summary(model)
    print(model.log_marginal_likelihood())
    print(model.kernel.lengthscales.unconstrained_variable)
main()
plt.show()
