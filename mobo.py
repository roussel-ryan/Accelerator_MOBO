import numpy as np
import pandas as pd

import logging
import time
import pickle

import pygmo as pg
import tensorflow as tf

from .multi_objective import pareto
from .multi_objective import td_pareto
from . import utilities as utils
from .multi_objective import plotting

from . import bo
from . import infill
from .optimizers import evolutionary
from . import trackers


class MultiObjectiveBayesianOptimizer(bo.BayesianOptimizer):
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
    
    def __init__(self, bounds, GPRs, B, **kwargs):
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

        self.GPRs         = GPRs
        self.B            = B

        acq       = kwargs.get('acq', infill.UHVI(1.0))
        opt       = kwargs.get('optimizer', evolutionary.SwarmOpt())

        super().__init__(bounds, opt, acq)

        self.A            = kwargs.get('A',np.zeros(self.obj_dim))
        #self.constraints  = kwargs.get('constraints',[])
        #self.constr_dim   = len(self.constraints)
        
        #self._use_constraints  = 1 if self.constr_dim > 0 else 0

        self.logger            = logging.getLogger(__name__)

        self._collect_gp_data()
        
    def _collect_gp_data(self):
        d = self._extract_data_from_GPRs()
        self._add_to_dataframe(*d)
        
        
    def get_obj_dim(self):
        return len(self.GPRs)

    def _add_to_GP(self, X, Y, C = None):
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
        npts = Y.shape[0]
        Y = Y.reshape(self.obj_dim, npts, 1)
    
        for i in range(self.obj_dim):
            #add observed data to GPRs
            y_data = Y[i]
            gpr = self.GPRs[i]
            gpr.data = (tf.concat((gpr.data[0],X),axis=0),
                        tf.concat((gpr.data[1],y_data),axis=0))

        #if self._use_constraints:
        #    C = C.reshape(self.constr_dim, npts, 1)
    
        #    for j in range(self.constr_dim):
        #        self.constraints[j].add_observations(X,C[j])

        self.PF = self.get_PF()
        
    def _get_optimization_stats(self):
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
     
        #measure distance travelled in input space
        x0 = self.GPRs[0].data[0][-1]
        #dist = np.linalg.norm(res.x - x0)

        
        #stats = {'n_pf':len(self.PF),
        #         'hypervolume':self.get_hypervolume(),
        #         'log_marginal_likelihood':[self.get_log_marginal_likelihood()]}
        stats = {}
        
        return stats

    def _extract_data_from_GPRs(self):
        X = self.GPRs[0].data[0].numpy()

        Y = []
        for i in range(len(self.GPRs)):
            Y += [self.GPRs[i].data[1].numpy()]

        Y = np.hstack(Y)
        return [X,Y]
            
    def get_PF(self):
        F = self.get_data('Y')
        return pareto.get_PF(F, self.B, tol = 1e-5)
        
    def get_hypervolume(self):
        hv = pg.hypervolume(self.get_PF())
        return hv.compute(self.B)

    def get_log_marginal_likelihood(self):
        res = np.array([ele.log_marginal_likelihood().numpy() for ele in self.GPRs])
        return res


class TDMultiObjectiveBayesianOptimizer(MultiObjectiveBayesianOptimizer,
                                        bo.TDOptimizer):
    ''' 
    time dependant multi-objective optimizer

    '''
    def __init__(self, bounds, GPRs, B, **kwargs):
        default_acq = infill.TDACQ(infill.NUHVI(gamma = 0.01))
        acq = kwargs.get('acq',default_acq)
        try:
            del kwargs['acq']
        except KeyError:
            pass
            
        MultiObjectiveBayesianOptimizer.__init__(self, bounds, GPRs, B,
                                                 acq = acq, **kwargs)
        bo.TDOptimizer.__init__(self)


    def get_data(self, name = 'all', **kwargs):
        #modifies get_data to only include measurements performed before "time"
        time = kwargs.get('time',self.time)
        try:
            del kwargs['time']
        except KeyError:
            pass
            
        t = super().get_data('t',**kwargs)
        ind = np.argwhere(t < time).flatten()[::2]
        
        data = super().get_data(name, **kwargs)
        return data[ind]
        
    def get_PCB_PF(self, **kwargs):
        return td_pareto.get_PCB_PF(self, **kwargs)

    
    def get_PCB_hv(self, **kwargs):
        PF = self.get_PCB_PF(**kwargs)
        if np.any(PF):
            hv = pg.hypervolume(PF)
            return hv.compute(self.B)
        else:
            return 0.0

    
    def _collect_gp_data(self):
        d = self._extract_data_from_GPRs()
        X = d[0]
        Y = d[1]
        self._add_to_dataframe(X[:,:-1], Y,
                               Z = {'t':X[:,-1].reshape(-1,1)})

    def _add_to_GP(self, X, Y, Z):
        X = np.hstack([X,Z['t']])
        for i in range(self.obj_dim):
            gpr = self.GPRs[i]
            self.GPRs[i].data = (tf.concat((gpr.data[0],X),axis=0),
                             tf.concat((gpr.data[1],Y[:,i].reshape(-1,1)),axis=0))

    def train(self, iters = 5000):
        for gpr in self.GPRs:
            self._train_hyp(gpr, iters)
    
    def print_model(self):
        for gpr in self.GPRs:
            self._print_model(gpr)

