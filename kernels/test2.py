import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary, set_trainable
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
import numdifftools as nd

import advanced_RBF
import utilities

dim = 10

def f(X,cov):
    #return multivariate_normal.pdf(X,mean=np.zeros(dim),cov=cov)
    return multivariate_normal.pdf(X,mean=np.zeros(dim),cov=cov)/multivariate_normal.pdf(np.zeros(dim),mean=np.zeros(dim),cov=cov)

def gfit(x,a,b):
    return a * np.exp(-0.5 * (x / b) ** 2)

def main():
    #specify bounds
    bounds = np.array((-5,5))
    
    #specify cov
    diag = False
    if diag:
        add = np.zeros(dim)
        add[::2] = 0.5
        d_ele = np.ones(dim) + np.arange(dim)*0.1 
        cov = np.diag(d_ele**2)

    else:
        cov = np.diag((np.ones(dim) + 0.1*np.arange(dim))**2) / 2
        cov += np.diag(0.1*np.arange(dim-1),k=1)
        cov += cov.T
        d_ele = np.ones(dim)
    print(cov)
    print(np.sqrt(cov)) 
    print(np.linalg.inv(cov))
    
    decomp = np.linalg.cholesky(np.linalg.inv(cov))
    L = decomp[np.triu_indices(dim)]
    print(L)

    
    #create training points
    dist = 'spherical'
    if dist == 'uniform':
        n_train = 5
        max_x = 4.5
        if dim > 6:
            x = [np.linspace(0.0,max_x,n_train) for i in range(dim)]
        else:
            x = [np.linspace(-max_x,max_x,n_train) for i in range(dim)]
        pts = np.meshgrid(*x)
        X = np.vstack([pts[i].ravel() for i in range(dim)]).T
        #X = np.vstack((X, np.zeros(dim)))
        print(f'training points: {len(X)}')
        print(f'{X[0:10]}')
        
    elif dist == 'random':
        n_train = 650
        X = np.random.uniform(*bounds,(n_train,dim))
        X = np.vstack((X,np.zeros(dim).reshape(-1,dim)))

    elif dist == 'spherical':
        n_r = 2
        n_theta = 2
        r_max = 5.0
        half_sphere = True
        X = utilities.generate_spherical_mesh(dim,n_r,n_theta,r_max,half_sphere)

    else:
        raise RuntimeError(f'dist \'{dist}\' not found')
        
    Y = f(X,cov).reshape(-1,1)
    #print(X)
    #print(Y)

    fig,ax = plt.subplots()
    #ax.pcolor(*pts,Y.reshape(n_train,n_train))
    
    base_kernel = gpflow.kernels.RBF(lengthscales=d_ele*np.sqrt(2))

    HP = {'L':L}
    test_kernel = advanced_RBF.AdvancedRBF(hyper_parameters = HP,
                                         mode = 'correlated')

    L_min = tf.cast(-5.0,dtype = tf.float64)
    L_max = tf.cast(5.0,dtype = tf.float64)

    test_kernel.L.transform = tfp.bijectors.Softplus(hinge_softness=0.0001)
    #test_kernel.L.transform = tfp.bijectors.Log()

    trans = tfp.bijectors.Sigmoid(low = L_min,high = L_max)
    #print(trans.__dict__)
    trans._dtype = tf.float64
    #test_kernel.L.transform = trans
    #base_kernel.lengthscales.transform = tfp.bijectors.Log()
    
    model = gpflow.models.GPR(data = (X,Y),
                              kernel = test_kernel)

    model.kernel.variance.transform = tfp.bijectors.Softplus(hinge_softness=0.0001)
    model.likelihood.variance.transform = None
    print(model.kernel.L)

    model.likelihood.variance.assign(1.0e-10)
    #model.kernel.variance.assign(f(np.zeros(dim),cov) * 1e-3 * np.pi**(dim/2) / np.linalg.det(cov)**0.5)
    model.kernel.variance.assign(0.1)

    #set_trainable(model.kernel.lengthscales,False)
    set_trainable(model.likelihood.variance,False)
    #set_trainable(model.kernel.variance,False)

    print_summary(model)

    #do optimization
    #model.kernel.lengthscales.assign((1.0,1.0))
    #model.kernel.L.assign((5.0,5.0))
    #opt = gpflow.optimizers.Scipy()
    opt = tf.optimizers.Adam(learning_rate=0.001)

    n_iter = 10000
    step = []
    log_likelihood = []
    lengthscales = []
    if model.kernel is test_kernel:
        for i in range(n_iter):
            if i % 100 ==0:
                print(f'{i}:{model.log_marginal_likelihood().numpy()}:{model.kernel.L.numpy()}')
                step.append(i)
                log_likelihood.append(model.log_marginal_likelihood().numpy())
                lengthscales.append(model.kernel.L.numpy())
            opt.minimize(model.training_loss,model.trainable_variables)

    if model.kernel is base_kernel:
        for i in range(n_iter):
            if i % 100 ==0:
                print(f'{i}:{model.log_marginal_likelihood().numpy()}:{model.kernel.lengthscales.numpy()}')
                step.append(i)
                log_likelihood.append(model.log_marginal_likelihood().numpy())
                lengthscales.append(model.kernel.lengthscales.numpy())
            opt.minimize(model.training_loss,model.trainable_variables)
            
    print_summary(model)
    #print(model.log_marginal_likelihood())
    #print(model.kernel.L)
    #print(model.kernel.upper_triangle)
    #print(model.kernel.get_covariance_matrix())
    #print(model.kernel.get_precision_matrix())
    
    #plot along x=0 line
    for j in range(dim):
        pts = np.zeros((100,dim))
        y = np.linspace(*bounds,100)
        #pts = np.vstack((*[np.zeros_like(y) for i in range(dim-1)],y)).T
        pts[:,j] = y
        F = f(pts,cov)
        F_pred = model.predict_f(pts)

        fig,ax = plt.subplots()
        ax.plot(y,F.ravel())
        ax.plot(y,F_pred[0])
        ax.plot(y,F_pred[0] + F_pred[1])
        ax.plot(y,F_pred[0] - F_pred[1])
    step = np.array(step)
    log_likelihood = np.array(log_likelihood)
    lengthscales = np.array(lengthscales).T
    fig,ax = plt.subplots()

    for i in range(len(lengthscales)):
        ax.plot(step,lengthscales[i])#,label=f'Ground truth lengthscale: {d_ele[i]}')
        
    ax.set_ylabel('lengthscale')
    ax.set_xlabel('step')
    ax.legend()
    fig.savefig('lengthscale_opt.svg')

    fig2, ax2 = plt.subplots(2,1)
    fit_cov = model.kernel.get_covariance_matrix().numpy()
    vmin = np.min(np.hstack((cov.ravel(),fit_cov.ravel())))
    vmax = np.max(np.hstack((cov.ravel(),fit_cov.ravel())))
    ax2[0].imshow(np.sqrt(2)*cov,vmin=vmin,vmax=vmax)
    ax2[1].imshow(fit_cov,vmin=vmin,vmax=vmax)
    
main()
plt.show()
