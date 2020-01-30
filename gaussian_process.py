# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:36:52 2020

@author: Ryan Roussel
"""
import logging


import numpy as np
from numba import jit
import matplotlib.pyplot as plt

from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
from scipy.spatial import distance

from sklearn.gaussian_process.kernels import StationaryKernelMixin 
from sklearn.gaussian_process.kernels import NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor


def _check_precision(X,precision):
    """
    Check to make sure that X and precision have the correct dimentions

    Args:
        X (TYPE): Input point.
        precision (TYPE): precision matrix.

    Returns:
        None.

    """
    pass

class AnisotropicRBF(StationaryKernelMixin, NormalizedKernelMixin,Kernel):
    """Custom Radial-basis function kernel (aka squared-exponential kernel).
    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. 
    
    Kernel is parameterized by the precision matrix, length scale, and sigma_f
    such that
    k(x,x') = exp(- (x - x').T * Precision * (x - x')/(2 l**2))
    
    
    Parameters
    ----------
    
    length_scale : float default: 1.0
        The length scale of the kernel.
    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """
    def __init__(self, length_scale = 1.0, length_scale_bounds = (1e-5,1e5),
                 precision_matrix = None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.precision_matrix = precision_matrix
        
    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)
    
    def __call__(self,X, Y=None, eval_gradient = False):
        """Return the kernel k(X, Y)
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """ 
        if not self.precision_matrix:
            precision = np.eye(X.shape[1])
        else:
            precision = self.precision_matrix
            
        logging.info(X)
        logging.info(Y)
        logging.info(precision)
            
        if Y is None:
            exp_term = - X @ precision @ X.T / (2*self.length_scale**2)
            K = np.exp(exp_term)
        else:
            exp_term = - (X - Y) @ precision @ (X - Y).T / (2*self.length_scale**2)
            K = np.exp(exp_term)
        return K
        

#define Gaussian (RBF) kernel with arbitrary precision matrix
class RBFKernel:
    """
    Gaussian RBF kernel that can handle arbitrary precision matricies

    Args:
        X1 (d x m array): Array of m points with dimention d.
        X2 (d x m array): Array of m points with dimention d.
        precision (TYPE, optional): Precision matrix with size (d x d) . 
            If not given, kernel is isotropic.
        l (TYPE, optional): Scalar length scale. Defaults to 1.0.
        sigma_f (TYPE, optional): Scalar height multiplier. Defaults to 1.0.

    Returns:
        Kernel value k(X1,X2).
    """
    def __init__(self,precision = None, sigma_f=1.0):
        #check if precision matrix is positive semi-definite
        if not np.all(np.linalg.eigvals(precision) > 0):
            raise ValueError(f'Precision matrix must be positive semi-definite:\
                             eignevals are {np.linalg.eigvals(precision)}')
        self.precision = precision
        self.sigma_f = sigma_f
    
    

    def __call__(self,X1,X2):    
        if not isinstance(self.precision,np.ndarray):
           precision = np.eye(X1.shape[1])
        else:
            precision = self.precision
        
        assert X1[0].shape[0] == precision.shape[0] and precision.shape[0] == precision.shape[1]
        precision = self.precision
        
        K = np.zeros((len(X1),len(X2)))
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                #K[i][j] = np.exp(- distance.mahalanobis(X1[i],X2[j],precision)**2)
                K[i][j] = np.exp(-(X1[i]-X2[j]).T @ precision @ (X1[i] - X2[j]))

        return self.sigma_f**2 * K

@jit
def _inv(X):
    return np.linalg.inv(X)
    
class GPRegressor:
    def __init__(self, kernel, sigma_y = 0.0):
        self.kernel = kernel
        self.sigma_y = sigma_y
        
    
    def train(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
        self.K = self.kernel(X_train, X_train) +\
            self.sigma_y**2 * np.eye(len(X_train))
        
    def predict(self, X_s, return_std=False):
        '''
        Computes the suffifient statistics of the GP posterior predictive distribution 
        from m training data X_train and Y_train and n new inputs X_s.
        
        Args:
            X_s: New input locations (d x n).
            X_train: Training locations (d x m).
            Y_train: Training targets (1 x m).
            l: Kernel length parameter.
            sigma_f: Kernel vertical variation parameter.
            sigma_y: Noise parameter.
        
        Returns:
            Posterior mean vector (n x d) and covariance matrix (n x n).
        '''
        
        K_s = self.kernel(self.X_train, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        
        
        K_inv = _inv(self.K)
        
        # Equation (4)
        mu_s = K_s.T.dot(K_inv).dot(self.Y_train)
    
        # Equation (5)
        if return_std:
            cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
            return mu_s, cov_s
        else:
            return mu_s

def main():
    #testing
    def f(x,precision):#test function with mean of (3,3)
        result = np.zeros((len(x)))
        x0 = np.array((1,1))
        for i in range(len(x)):
            result[i] = np.exp(- distance.mahalanobis(x[i],x0,precision)**2)
        return result
    
    cov = np.array(((1,-1/2),(-1/2,1)))    
    
    bounds = [-3,3]
    n = 20
    x = np.linspace(*bounds,n)
    y = np.linspace(*bounds,n)
    
    xx,yy = np.meshgrid(x,y)
    
    pts = np.vstack((xx.ravel(),yy.ravel())).T
    #logging.info(pts)
    ground_truth = f(pts,cov)
    
    fig,ax = plt.subplots()
    ax.contourf(xx,yy,ground_truth.reshape(n,n),cmap='Blues')
    
    dim = 2
    n_pts = 15
    X_train = np.random.uniform(*bounds,size = (n_pts,dim))
    #X_train = np.array(((3,3),(2,2),(0,0)))
    Y_train = f(X_train,cov)
    
    
    k_correlated = RBFKernel(precision=cov)
    k_uncorrelated = RBFKernel(precision=np.eye(2))   

    gprc = GPRegressor(k_correlated)
    gpru = GPRegressor(k_uncorrelated)
    #logging.info(pts)
    #logging.info(X_train)
    #logging.info(Y_train)
    gprc.train(X_train,Y_train)
    gpru.train(X_train,Y_train)
   
    muc = gprc.predict(pts)
    muu = gpru.predict(pts)
    
    #logging.info(mu)
    fig2,ax2 = plt.subplots(3,1)
    ax2[0].contourf(xx,yy,ground_truth.reshape(n,n),cmap='Blues')
    ax2[1].contourf(xx,yy,muc.reshape(n,n),cmap='Blues')
    ax2[2].contourf(xx,yy,muu.reshape(n,n),cmap='Blues')
    
    for ax in ax2:
        for ele in X_train:
            ax.plot(*ele,'+r')
    
if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
