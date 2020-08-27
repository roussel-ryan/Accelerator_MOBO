import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy import special

class Constraint:
    '''
    Object to store a single constraint GaussianProcessRegressor

    If a constraint is satified then the predict function returns one, 
    if the constraint is not satisfied then it must return zero.

    For consistency the GP itself does not model ths zero - one range, it models
    the function g(x). The predict method returns norm.CDF(h,mu(x),sigma(x)) which is 
    the probability that the function value satisfies g(x) <= h

    See the following reference for more info 
    Gardner, Jacob R., et al. "Bayesian Optimization with Inequality Constraints." 
    ICML. Vol. 2014. 2014.
    
    Attributes
    ----------
    GPR : gpflow.model
        Gaussian process surrogate model

    h : float
        Constant h, see above

    '''

    def __init__(self, GPR, h, invert = False):
        self.GPR   = GPR
        self.h     = h

        self.invert = invert


        
    def add_observations(self,X,C):
        '''add observation of constraint function

        Parameters:
        -----------
        X : ndarray, shape (n, input_dim)
            Independant variable location
        
        C : ndarray, shape (n,1)
            Observed constraint value

        Returns:
        --------
        None

        '''
        
        self.GPR.data = (tf.concat((self.GPR.data[0], X), axis = 0),
                         tf.concat((self.GPR.data[1], C), axis = 0))

    def get_feasable(self):
        ''' return a boolean matrix showing where stored points are feasable'''
        C = self.GPR.data[1].numpy().flatten()

        b = np.where(C < self.h, 1, 0)
        if self.invert:
            return not b
        else:
            return b 
        
        
    def predict(self,X):
        mu, sig = self.GPR.predict_f(X)

        inversion_mult = 1
        if self.invert:
            inversion_mult = -1
        
        return 0.5 * (1 + special.erf(inversion_mult * (self.h - mu) / (np.sqrt(2 * sig))))


