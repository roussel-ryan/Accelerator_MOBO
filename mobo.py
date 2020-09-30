import numpy as np
import pandas as pd

import logging
import time
import pickle

import pygmo as pg
import tensorflow as tf

from .multi_objective import EHVI
from .multi_objective import pareto
from .multi_objective import utilities as utils
from .multi_objective import plotting

from . import infill
from . import optimizers
from . import trackers


class MultiObjectiveBayesianOptimizer:
    """Multiobjective bayesian optimizer

    This class impliments a m-D multi-objective Bayesian optimizer
    which uses m Gaussian Processes/kriging models 
    (one for each objective) to predict points in the n-D input 
    space that will maximize the Truncated Expected 
    Hypervolume Improvement (TEHVI).

    This class uses a LineOpt instance to maximize the TEHVI
    acquisition function as it is efficient in high dimentional
    input spaces

    Attributes
    ----------
    bounds : sequence
        Sequence of (min,max) pairs for each independant variable

    GPRs : list
        List of GPFlow model objects
        (one for each independant variable).

    X : ndarray, shape (p,n)
        Array of p observed input point locations.

    F : ndarray, shape (p,m)
        Array of p observed objective function values.

    B : ndarray, shape (m,)
        Upper bound of objective space, also referred to as
        the reference point.

    input_dim : int
        Dimentionality of input space, equal to n.
    
    obj_dim : int
        Dimentionality of output space, equal to m.

    A : ndarray, shape (m,)
        Lower bound of objective space.

    constraints : list
        List of Constraint objects that represent constraint 
        functions on the inputs space

    """
    
    def __init__(self,bounds,GPRs,B,**kwargs):
        """ Initialization
        
        Parameters:
        -----------

        bounds : sequence
            Sequence of (min,max) pairs for each independant 
            variable
        
        GPRs : list
            List of GPFlow models
            (one for each independant variable).

        B : ndarray, shape (m,)
            Upper bound of objective space, also referred to as
        the reference point.

        A : ndarray, shape (m,), optional
            Lower bound of objective space. 
            Default: np.zeros(obj_dim)

        constraints : list, optional
            List of Constraint objects that represent constraint 
            functions on the inputs space. Defualt: []
        
        verbose : bool, optional
            Display diagnostic plots. Default: False

        """

        self.bounds       = bounds
        self.GPRs         = GPRs
        self.B            = B

        self.input_dim    = len(self.bounds)
        self.obj_dim      = len(self.GPRs)
        
        self.A            = kwargs.get('A',np.zeros(self.obj_dim))
        self.constraints  = kwargs.get('constraints',[])
        self.constr_dim   = len(self.constraints)
        self.verbose      = kwargs.get('verbose',False)

        self._use_constraints  = 1 if self.constr_dim > 0 else 0

        self.logger            = logging.getLogger(__name__)

        
        #set infill function (defualt UHVI w/ beta = 1.0)
        self.infill                 = kwargs.get('infill',
                                                 infill.UHVI(1.0))
        
        #add constraints if necessary
        if self._use_constraints:
            self.logger.info('Using constraint function')
            self.obj = self.constrained_infill
        else:
            self.obj = self.infill

            
        #create a pandas dataframe to store info about observations
        self.update_model_data()
        self.logger.info(self.data)
        self.PF = self.get_PF()

        self.t = 0
        self.history = []

    def update_model_data(self):
        self.data = utils.create_gp_dataframe(self)
        
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

        #reshape Y from (n, output_dim) -> (output_dim, n, 1) to properly stack
        #do the same for C
        npts = Y.shape[0]
        Y = Y.reshape(self.obj_dim, npts, 1)
    
        for i in range(self.obj_dim):
            #add observed data to GPRs
            y_data = Y[i]
            gpr = self.GPRs[i]
            gpr.data = (tf.concat((gpr.data[0],X),axis=0),
                        tf.concat((gpr.data[1],y_data),axis=0))

        if self._use_constraints:
            C = C.reshape(self.constr_dim, npts, 1)
    
            for j in range(self.constr_dim):
                self.constraints[j].add_observations(X,C[j])

        self.update_model_data()
        self.logger.info(f'\n{self.data}')

        self.PF = self.get_PF()
        
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
        
        #do optimization step to maximize obj (minimize neg_obj)
        args = [self.GPRs,self.PF,self.A,self.B]
        def _neg_obj(x, *args):
            return -1.0 * self.obj(x, *args) 

        start = time.time()
        self.logger.info('Starting acquisition function optimization')
        #self.logger.info(f'UHVI beta {self.infill.get_beta()}')

        res = optimizer_func(self.bounds, _neg_obj, args)
        exec_time = time.time() - start
        self.logger.info(f'Done with optimization in {exec_time} s')
        #self.logger.info(f'Avg UHVI calc time {self.infill.get_avg_time()} s')
        #self.infill.reset_timer()

        #measure distance travelled in input space
        x0 = self.GPRs[0].data[0][-1]
        dist = np.linalg.norm(res.x - x0)

        
        stats = pd.DataFrame({'exec_time':exec_time,
                              'n_obs':self.n_observations,
                              'n_iterations' : self.t,
                              'n_valid' : len(self.get_data('X',valid=True)),
                              'n_pf':len(self.PF),
                              'dist':dist,
                              'hypervolume':self.get_hypervolume(),
                              'predicted_ideal_point':[res.x],
                              'predicted_hypervolume_improvment':np.abs(res.f),
                              'actual_hypervolume_improvement':np.nan,
                              'log_marginal_likelihood':[self.get_log_marginal_likelihood()]},
                             index = [self.t])

        if isinstance(self.infill,infill.UHVI):
            stats['beta'] = self.infill.get_beta()
            
        if isinstance(self.history,pd.DataFrame):
            a = stats['hypervolume'] - self.history.at[self.t-1,'hypervolume'] 
            self.history.at[self.t-1,'actual_hypervolume_improvement'] = a
            self.history = pd.concat([self.history,stats])
        else:
            self.history = stats

        self.t += 1

        return res

    def save(self, fname):
        pickle.dump(self,open(fname,'wb'))
    
    def get_data(self, name = 'all', valid = None, convert = True):
        return utils.get_data(self, name, valid, convert)
    
    def get_PF(self):
        F = self.get_data('Y', valid = True)
        return pareto.get_PF(F, self.B, tol = 1e-5)
        
    def get_hypervolume(self):
        hv = pg.hypervolume(self.get_PF())
        return hv.compute(self.B)

    def get_log_marginal_likelihood(self):
        res = np.array([ele.log_marginal_likelihood().numpy() for ele in self.GPRs])
        return res

    def plot_acq(self,ax = None, **kwargs):
        return plotting.plot_acq(self,ax, **kwargs)

    def plot_constr(self,ax = None,**kwargs):
        return plotting.plot_constr(self,ax,**kwargs)

    def constrained_infill(self, x, *args):
        cval = np.array([ele.predict(
            x.reshape(-1,self.input_dim)) for ele in self.constraints])
        constr_val = np.prod(cval)
        self.constraint_vals = cval
        return self.infill(x,*args) * constr_val
