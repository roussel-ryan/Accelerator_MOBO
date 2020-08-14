import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import time

import pygmo as pg
import tensorflow as tf

from .multi_objective import EHVI

from . import infill
from . import optimizers
from . import trackers

class SingleObjectiveBayesianOptimizer:
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

        self.bounds       = bounds
        self.GPR          = GPR

        self.input_dim    = len(self.bounds)
        self.constraints  = kwargs.get('constraints',[])
        self.verbose      = kwargs.get('verbose',False)

        self.n_constr          = len(self.constraints)
        self._use_constraints  = 1 if self.n_constr > 0 else 0

        self.logger            = logging.getLogger(__name__)

        
        #set infill function (defualt UHVI w/ beta = 1.0)
        self.infill                 = kwargs.get('infill',
                                                 infill.UCB())
        
        #add constraints if necessary
        if not self._use_constraints:
            self.obj = self.infill
        else:
            self.obj = self.constrained_infill
        

        self.history = []
        
        
    def add_observations(self, X, Y, C = None):
        '''add observed data to gaussian process regressors

        Parameters
        ----------
        X : ndarray, shape (n, input_dim)
            Observed input points to add

        Y : ndarray, shape (n, obj_dim)
            Observed output points to add

        C : ndarray, shape (n, constraint_dim) , optional
            Constraint observation data to add

        Returns
        -------
        None

        '''
        self.logger.debug(f'adding observation(s) X:{X}, Y:{Y}, C:{C}')

        
        y_data = Y
        self.GPR.data = (tf.concat((self.GPR.data[0],X),axis=0),
                         tf.concat((self.GPR.data[1],y_data),axis=0))

        #TODO: add constraint data acceptance
        
    def get_next_point(self, optimizer_func, **kwargs):
        '''get the point that optimizes TEHVI acq function

        Parameters:
        -----------
        optimizer_func : callable 
            Function call used to optimize TEHVI.
        
        return_value : bool
            Whether or not to return the function value f(x*)

        **kwargs are used as arguments to optimizer_func

        Returns:
        --------
        x* : ndarray, shape (n,)
            Input value that maximized TEHVI

        f* : float
            Acquisition function value at x*, 
            if return_value == True

        '''
        
        t = self.infill.t    

        start = time.time()
        self.logger.info(f'Starting acquisition function optimization step {self.infill.t}')
        #self.logger.info(f'UHVI beta {self.infill.get_beta()}')

        args = self.GPR
        res = optimizer_func(self.bounds, self._neg_obj, args)

        exec_time = time.time() - start
        self.logger.info(f'Done with optimization in {exec_time} s')

        stats = pd.DataFrame({'exec_time':exec_time,
                              'n_obs':len(self.GPR.data[1]),
                              'predicted_ideal_point':[res.x],
                              'log_marginal_likelihood':[self.get_log_marginal_likelihood()]},
                             index = [t])

        if isinstance(self.infill,infill.UHVI):
            stats['beta'] = self.infill.get_beta()
        
        #if isinstance(self.history,pd.DataFrame):
            #self.history.at[t-1,'actual_hypervolume_improvement'] = stats['hypervolume'] - self.history.at[t-1,'hypervolume'] 
            #self.history = pd.concat([self.history,stats])
        #else:
        #    self.history = stats
        
        self.infill.t += 1

        return res
    
    def _neg_obj(self, x, *args):
        return -1.0 * self.obj(x, *args) 

    
    def get_log_marginal_likelihood(self):
        return self.GPR.log_marginal_likelihood().numpy()   
            
            
    def constrained_infill(self, x, *args):
        cval = np.array([ele.predict(x.reshape(-1,self.input_dim)) for ele in self.constraints])
        constr_val = np.prod(cval)
        self.constraint_vals = cval
        return self.infill(x,*args) * constr_val


    

        
