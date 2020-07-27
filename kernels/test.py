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

tf.compat.v1.enable_eager_execution()

DIM = 10

def get_tri_elements(M,dim):
    triu_idx = np.triu_indices(dim,k = 0)
    elements = M[triu_idx]
    return elements

def f(X,precision):
    #k = tf.shape(X)[1].numpy()
    #y = X @ precision @ tf.transpose(X)
    #Z = (2 * np.pi)**(0.5*k) * tf.linalg.det(tf.linalg.inv(precision))**0.5
    #return tf.exp(-0.5*y) / Z
#    return multivariate_normal.pdf(X,mean=np.zeros(DIM),cov=np.linalg.inv(precision))
    return multivariate_normal.pdf(X,mean=np.zeros(DIM),cov=precision)

def test_plot(precision):
    fig,ax = plt.subplots()
    x = np.linspace(-5,5).reshape(-1,1)

def main():
    precision = np.diag(np.arange(1,DIM+1))
    #for i in range(1,3):
    #    offdiag = np.random.uniform(-0.5,0.5,DIM - i)
    #    precision = precision + np.diag(offdiag,k = i) + np.diag(offdiag,k = -i)
    

    #print(precision)
    #test_plot(precision)

    n_test = 3**DIM
    print(f'n_test {n_test}')
    x = [np.linspace(-0.75,0.75,int(n_test**(1/DIM))) for i in range(DIM)]
    xx = np.meshgrid(*x)
    pts = np.vstack([ele.ravel() for ele in xx]).T
    F = f(pts,precision).reshape(-1,1)

    #print(pts.shape)
    
    #hessian
    #print(f'Function hessian at origin {nd.Hessian(f)(np.zeros(DIM))}')
    #hessian = -nd.Hessian(f)(np.zeros(DIM))/(2*f(np.zeros(DIM)))
    #print(f'proposed precision matrix: \n{hessian}')
    #L = get_tri_elements(np.linalg.cholesky(hessian).T,DIM)
    #print(f'decomposed matrix \n{np.linalg.cholesky(hessian).T}')

    dim = DIM
    n_training = 500

    print(f'n_train: {n_test}')

    if dim == 1:
        X = np.linspace(-5,5,n_training).reshape(-1,1)
    else:
        X = np.random.uniform(-5,5,(n_training,dim))
    Y = f(X,precision).reshape(-1,1)


    
    #k = gpflow.kernels.RBF()

    #----------------------------
    #define kernel
    mode = 'anisotropic'
    #mode = None
    
    if mode == 'anisotropic':
        L = np.arange(1,DIM+1)
        #L = np.ones(DIM)


    HP = {'L':L}
    corr = correlated.AdvancedRBF(hyper_parameters = HP,mode=mode)
    corr.L.transform = tfp.bijectors.Softplus()
    #corr.L.transform = tfp.bijectors.Chain((tfp.bijectors.Log(),tfp.bijectors.Softplus()))
    corr.variance.transform = tfp.bijectors.Softplus()
    corr.variance.assign(0.001)
    #set_trainable(corr.variance,False)
    
    test_kernel = corr
    #model = gpflow.models.GPR(data = (X,Y),kernel=test_kernel,mean_function=None)
    model = gpflow.models.GPR(data = (X,Y), kernel = gpflow.kernels.RBF(variance=0.001,lengthscales=np.ones(DIM)))
    
    model.likelihood.variance.assign(1e-5)
    set_trainable(model.likelihood.variance,False)
    set_trainable(model.kernel.variance,False)
    
    print_summary(model)
    print(model.log_marginal_likelihood())

    
    #opt = gpflow.optimizers.Scipy()
    opt = tf.optimizers.Adam(learning_rate = 0.01) 
    #opt = tf.optimizers.SGD()
    print('doing minimization')

#    print(model.predict_f(np.array((-1)).reshape(-1,1)))
    
    monitor_freq = 50
    iterations = 1000
    log_mlk = np.zeros(int(iterations / monitor_freq))

    for i in range(iterations):
        if i % monitor_freq == 0:
            print(i)
            log_mlk[int(i / monitor_freq)] = model.log_marginal_likelihood()
        opt.minimize(model.training_loss, model.trainable_variables)
        #base_opt.minimize(m.training_loss, m.trainable_variables)



    print_summary(model)
    print(model.log_marginal_likelihood())

    
    #mode = None
    
    if mode == 'correlated':
        L = model.kernel.L
        tri = model.kernel.construct_upper_triangle(L)
        precision_fit = tf.linalg.matmul(tri,tf.transpose(tri)).numpy()
        print(tf.linalg.matmul(tri,
                               tf.transpose(tri)).numpy())
        print(tri.numpy())

    if mode == 'anisotropic':
        L = model.kernel.L.numpy()
        print(model.kernel.L)
        precision_fit = np.diag(L)

    #fig,ax = plt.subplots(3,1)

    fig,ax = plt.subplots()
    ax.plot(log_mlk)

    #plot precision matricies
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(precision)
    ax[1].imshow(precision_fit)

    if DIM == 1:
        fig,ax = plt.subplots()
        x = np.linspace(-10,10).reshape(-1,1)
        F = f(x,precision).reshape(-1,1)
    
        p = model.predict_f(x)
        m = p[0].numpy().ravel()
        s = p[1].numpy().ravel()
        x = x.ravel()
        ax.plot(x,m)
        ax.fill_between(x,m-s,m+s,lw=0,alpha=0.25)
        ax.plot(x,F)
        ax.plot(X,Y,'r+')


main()
plt.show()
