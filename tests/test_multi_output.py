import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import time

import gpflow
from gpflow.utilities import print_summary

def f1(X):
    return np.sin(X)

def f2(X):
    return 1.5 * np.sin(X + np.pi/4.)

def fn(X,A,B):
    return A * np.sin(X + B)

def generate_data(bounds, P):
    
    #initial points
    N = 100
    X = np.linspace(*bounds,N).reshape(-1,1)

    Y = np.hstack((f1(X),f2(X)))

    #add random functions
    for i in range(2,P):
        A = np.random.uniform(0.5,1.5)
        B = np.random.uniform(-np.pi/2, np.pi/2)
        Y = np.hstack((Y,fn(X,A,B)))
    
    return X, Y


def optimize_model_with_scipy(model,data):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": 1000},
    )

def main():
    bounds = np.array((-1,1))*np.pi
    P = 7
    
    X, Y = data = generate_data(bounds,P)

    
    
    fig,ax = plt.subplots()
    for ele in Y.T:
        ax.plot(X,ele,'+')


    #prediction points
    Z = np.linspace(*bounds*2).reshape(-1,1)
    
    #create kernels
    kernels = [gpflow.kernels.RBF(lengthscales = 0.5, variance = 0.5) for _ in range(P)]
    kernel = gpflow.kernels.SeparateIndependent(kernels)

    #create inducing variables
    X0 = np.linspace(*bounds,10).reshape(-1,1)
    iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(X0))

    
    
    m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps = P)
    #m = gpflow.models.GPR(data,kernel = kernel, noise_variance = 0.0001)
    
    
    optimize_model_with_scipy(m,data)
    
    print_summary(m)


    
    
    means, variances = m.predict_f(Z)
    means = means.numpy()
    variances = np.sqrt(variances.numpy())

    print(variances)
    
    for mu,std,c in zip(means.T,variances.T,range(P)):
        ax.plot(Z,mu,c=f'C{c}')
        ax.fill_between(Z.flatten(), mu - std, mu + std, color = f'C{c}', alpha = 0.25, lw=0)
    

    #evaluation timing
    start = time.time()
    m.predict_f(np.array((1.0)).reshape(-1,1))
    print(time.time() - start)
        
    
    #print(m.predict_f(Z))

    
main()
plt.show()

    
    
