import numpy as np
import matplotlib.pyplot as plt

import logging
import time

from .multi_objective import pareto
from .multi_objective import EHVI

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
        self.verbose      = kwargs.get('verbose',False)

        self.n_constr          = len(self.constraints)
        self._use_constraints  = 1 if self.n_constr > 0 else 0
        
        self.normalize_input   = kwargs.get('norm_input',False)
        self.normalize_obj     = kwargs.get('norm_obj',False)

        self.n_restarts        = kwargs.get('n_restarts',10)
        self.optimization_freq = kwargs.get('optimization_freq',-1)

        self.n_refits          = 0

        self.logger            = logging.getLogger(__name__)

        self.history           = []
        
    def fit(self, X, F, C=None):
        #update observation data
        if self.normalize_input:
            X = self._norm_input(X)

        if self.normalize_obj:
            F = self._norm_obj(F)
          
        self.X = X
        self.F = F
        self.C = C

        tracker = trackers.GPTracker(self.input_dim,len(self.X))
        #train objective function GPs

        for i in range(self.obj_dim):
            self.GPRs[i].data((self.X,self.F[:,i].reshape(-1,1)))

            if self.optimization_freq == -1: 
                self.GPRs[i].optimize_restarts(
                    num_restarts = self.n_restarts,
                    max_f_eval = 1000)
                tracker.stats['optimization'] = True
                tracker.stats['n_restarts'] = self.n_restarts
                
            else:
                if self.n_refits % self.optimization_freq == 0:
                    self.GPRs[i].optimize_restarts(
                        num_restarts = self.n_restarts,
                        max_f_eval = 1000)
                    tracker.did_optimization = True

            tracker.models.append(self.GPRs[i].to_dict())
                    
        #train constraint function GPs
        if self._use_constraints:
            for j in range(self.n_constr):
                self.constraints[j].fit(X, C[:,j].reshape(-1,1))

        tracker.end()

        self.history.append(tracker)
        self.n_refits += 1
            
    def get_next_point(self, optimizer, return_value = False):
        '''get the point that optimizes TEHVI acq function

        Parameters:
        -----------
        optimizer : BlackBoxOptimizer
            Instance of BlackBoxOptimizer used to optimize TEHVI.
        
        return_value : bool
            Whether or not to return the function value f(x*)
        
        Returns:
        --------
        x* : ndarray, shape (n,)
            Input value that maximized TEHVI

        f* : float
            Acquisition function value at x*, 
            if return_value == True

        '''
        if self.verbose and self.input_dim == 2:
            self.plot_acq()

        if self.obj_dim == 2:
            self.PF = pareto.get_PF(self.F) 
            self.PF = pareto.sort_along_first_axis(self.PF)

            if self.normalize_obj:
                fargs = [self.GPRs,self.PF,
                         np.ones(self.obj_dim),
                         np.zeros(self.obj_dim)]
            else:
                fargs = [self.GPRs,self.PF,self.B,self.A]

            if self.normalize_input:
                bounds = np.vstack((np.zeros(self.input_dim),
                                    np.ones(self.input_dim))).T
            else:
                bounds = self.bounds
                
            x0 = self.X[-1]

            if not self._use_constraints:
                obj = EHVI.get_EHVI
            else:
                obj = self._constr_EHVI

            if isinstance(optimizer,optimizers.ParallelLineOpt):
                #get PF x values as a replacement for x0
                PF_indicies = pareto.get_PF_indicies(self.F)
                x0 = self.X[PF_indicies]
                
            res = optimizer.minimize(bounds,
                                     obj,
                                     args = fargs,
                                     x0 = x0)
        #undo normalization
        if self.normalize_input:
            res.x = self._denorm_input(res.x)

        if self.normalize_obj:
            res.f = self._denorm_obj(res.f)
            
        if return_value:
            return res.x, res.f
        else:
            return res.x

    def plot_acq(self,ax = None):
        if ax is None:
            new_plot = True
        
        if new_plot:
            fig,ax = plt.subplots(self.obj_dim + 1,1)
        else:
            fig,ax = plt.subplots()

        self.PF = pareto.get_PF(self.F) 
        self.PF = pareto.sort_along_first_axis(self.PF)

        fargs = [self.GPRs,self.PF,self.B,self.A]    

        n = 20
        x = np.linspace(*self.bounds[0,:],n)
        y = np.linspace(*self.bounds[1,:],n)
        xx, yy = np.meshgrid(x,y)
        pts = np.vstack((xx.ravel(),yy.ravel())).T

        f = []
        for i in range(n**2):
            if not self._use_constraints:
                f.append(EHVI.get_EHVI(pts[i],*fargs))
            else:
                f.append(self._constr_EHVI(pts[i],*fargs))
        f = np.array(f).reshape(n,n)

        if new_plot:
            #plot all of the obj function predictions
            for j in range(self.obj_dim):
                c = ax[j].pcolor(xx,
                                 yy,
                                 self.GPRs[j].predict(pts)[0].reshape(n,n))
                ax[j].figure.colorbar(c,ax=ax[j])
                
            c = ax[self.obj_dim].pcolor(xx,yy,f)
            ax[self.obj_dim].figure.colorbar(c,
                                             ax= ax[self.obj_dim])
            for a in ax:
                a.plot(*self.X.T,'r+')
                
                
            fig.suptitle(f'MOBO problem step {len(self.X)}')
        else:
            c = ax.pcolor(xx,yy,f)
            ax.figure.colorbar(c,ax=ax)          

        
            
    def _constr_EHVI(self,x,*args):
        cval = np.array([ele.predict(x.reshape(-1,self.input_dim)) for ele in self.constraints])
        constr_val = np.prod(cval)
        self.constraint_vals = cval
        return EHVI.get_EHVI(x,*args) * constr_val

    def _norm_input(self,X):
        logging.info(self.bounds[:,1].T)
        width = self.bounds[:,1] - self.bounds[:,0]
        return (X - self.bounds[:,0]) / width.T

    def _denorm_input(self,X):
        width = self.bounds[:,1].T - self.bounds[:,0].T
        return X * width.T + self.bounds[:,0].T

    
    def _norm_obj(self,X):
        width = self.B - self.A
        return (X - self.A) / width.T

    def _denorm_obj(self,X):
        width = self.B - self.A
        return X * width.T + self.A


    

        
