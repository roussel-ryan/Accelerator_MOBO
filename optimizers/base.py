import numpy as np

import logging

from .lineBO import lineOpt

class Result:
    '''Result class

    Mirrors scipy.minimize result class


    '''
    def __init__(self,x,f):
        self.x = x
        self.f = f


class BlackBoxOptimizer:
    '''BlackBoxOptimizer class 

    Class that should be subclassed by optimizer objects
    '''
    def __init__(self):
        pass

    def minimize(self, bounds, func, args = [], x0 = None):
        raise NotImplementedError



             
