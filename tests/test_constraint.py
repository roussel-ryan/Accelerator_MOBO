import gpflow
import numpy as np
import matplotlib.pyplot as plt

from GaussianProcessTools import constraints

def f(x):
    return x**2

def main():
    
    #sample the objective functions
    n_initial = 4
    X0 = np.random.uniform(-2,2,size = (n_initial,1))
    Y0 = f(X0)

    X = X0
    Y = Y0
    
    
    #define kernels to be used for the gaussian process regressors
    kernel = gpflow.kernels.RBF(lengthscales = 0.5, variance = 0.5)

    GPR = gpflow.models.GPR((X,Y),kernel, noise_variance = 0.0001)

    constraint = constraints.Constraint(GPR,1,True)
    

    x = np.linspace(-5,5,100).reshape(-1,1)

    const_satisfied = constraint.predict(x)
    
    fig,ax = plt.subplots()
    mu,var = GPR.predict_f(x)

    mu = mu.numpy().flatten()
    sig = np.sqrt(var.numpy().flatten())
    ax.plot(x,mu)
    ax.fill_between(x.flatten(),mu-sig,mu+sig, alpha=0.25, lw = 0)

    ax.plot(x,const_satisfied)
    
main()
plt.show()
