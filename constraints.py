import logging
import numpy as np
import matplotlib.pyplot as plt

class Constraint:
    '''
    Object to store a single constraint GaussianProcessRegressor

    GPR           - GPy GPRegressor object
    cfunction     - constraint function of the form f(x,*args) that returns 0 or 1 if constraint is unsatisfied/satisfied
    args          - external arguments to cfunction
 
    '''

    def __init__(self,GPR,cfunction,args=[]):
        self.GPR          = GPR
        self.cfunction    = cfunction
        self.args         = args
        
    def fit(self,X,C):
        #train gaussian process regressor
        self.X_train = X
        self.C = C
        self.C_train = self.cfunction(self.C,*self.args)
        self.GPR.set_XY(self.X_train,self.C_train)
        self.GPR.optimize()
        
    def predict(self,X):
        return self.GPR.predict(X)
