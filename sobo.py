import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import time

import tensorflow as tf

from . import infill
from .optimizers import evolutionary
from . import bo

class SingleObjectiveBayesianOptimizer(bo.BayesianOptimizer):
    """SingleObjective bayesian optimizer

    This class impliments a single-objective Bayesian optimizer
    which uses a Gaussian Processes/kriging model 
    to predict points in the n-D input 
    space that will maximize the Truncated Expected 
    Hypervolume Improvement (TEHVI).

    This class uses a LineOpt instance to maximize the TEHVI
    acquisition function as it is efficient in high dimentional
    input spaces

    Attributes
    ----------
    bounds : sequence
        Sequence of (min,max) pairs for each independant variable

    GPR : list
        GPFlow GPR model

    X : ndarray, shape (p,n)
        Array of p observed input point locations.

    F : ndarray, shape (p,1)
        Array of p observed objective function values.

    input_dim : int
        Dimentionality of input space, equal to n.
    
    constraints : list
        List of Constraint objects that represent constraint 
        functions on the inputs space

    """
    
    def __init__(self, bounds, GPR, **kwargs):
        """ Initialization
        
        Parameters:
        -----------

        bounds : sequence
            List of (min,max) pairs for each independant 
            variable
        
        GPR : GPflow.model
            GPFlow model.

        constraints : list, optional
            List of Constraint objects that represent constraint 
            functions on the inputs space. Defualt: []
        
        verbose : bool, optional
            Display diagnostic plots. Default: False

        """

        
        
        self.GPR          = GPR        
        self.logger       = logging.getLogger(__name__)

        
        acq               = kwargs.get('acq', infill.UCB())
        optimizer         = kwargs.get('optimizer', evolutionary.SwarmOpt())

        self.logger.info(f'Using acquisition function {acq} \n' +\
                         f'with settings {acq.settings}')
        
        super().__init__(bounds, optimizer, acq)

        self._collect_gp_data()
        
    def _collect_gp_data(self):
        self._add_to_dataframe(self.GPR.data[0],
                               self.GPR.data[1])
        
    def get_obj_dim(self):
        return 1
        
    def _add_to_GP(self, X, Y, Z):
        gpr = self.GPR
        self.GPR.data = (tf.concat((gpr.data[0],X),axis=0),
                    tf.concat((gpr.data[1],Y),axis=0))

    def train(self, iters = 5000):
        self._train_hyp(self.GPR, iters)

    def print_model(self):
        self._print_model(self.GPR)
        

        
class TDSingleObjectiveBayesianOptimizer(SingleObjectiveBayesianOptimizer,
                                         bo.TDOptimizer):
    '''
    we assume that the last input data axis is time, 
    we modify the optimization s.t. time axis is not optimized

    NOTE: due to this the assertions about input dim size will go to false!
    
    '''
    def __init__(self, bounds, GPR, **kwargs):
        self.GPR          = GPR        
        self.logger       = logging.getLogger(__name__)

        default_acq       = infill.TDACQ(infill.UCB(maximize = False))
        acq               = kwargs.get('acq', default_acq)
        del kwargs['acq']
        
        SingleObjectiveBayesianOptimizer.__init__(self, bounds, GPR,
                                                  acq = acq, **kwargs)
        bo.TDOptimizer.__init__(self)        
        
    def _collect_gp_data(self):
        self._add_to_dataframe(self.GPR.data[0][:,:-1],
                               self.GPR.data[1],
                               {'t':self.GPR.data[0][:,-1].numpy().reshape(-1,1)})

        
    def get_obj_dim(self):
        return 1

    def _add_to_GP(self, X, Y, Z):
        gpr = self.GPR
        X = np.hstack([X,Z['t']])
        self.GPR.data = (tf.concat((gpr.data[0],X),axis=0),
                    tf.concat((gpr.data[1],Y),axis=0))

    def train(self, iters = 5000):
        self._train_hyp(self.GPR, iters)
    
    def print_model(self):
        self._print_model(self.GPR)
        
        
