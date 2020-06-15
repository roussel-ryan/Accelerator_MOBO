import numpy as np
import matplotlib.pyplot as plt

import logging

from . import pareto
from . import EHVI

from .. import optimizers

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
        List of scikit_learn GaussianProcessRegressor objects
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
            List of GPy GPRegression models
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

        self.n_constr     = len(self.constraints)
        self._use_constraints = 1 if self.n_constr > 0 else 0
        
    
    def fit(self, X, F, C=None):
        #update observation data
        self.X = X
        self.F = F
        self.C = C

        #train objective function GPs
        for i in range(self.obj_dim):
            self.GPRs[i].set_XY(X = self.X,Y = self.F[:,i].reshape(-1,1))
            self.GPRs[i].optimize(max_f_eval = 1000)

        #train constraint function GPs
        if self._use_constraints:
            for j in range(self.n_constr):
                self.constraints[j].fit(X, C[:,j].reshape(-1,1))
            
    
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
        if self.obj_dim == 2:
            self.PF = pareto.get_PF(self.F) 
            self.PF = pareto.sort_along_first_axis(self.PF)

            fargs = [self.GPRs,self.PF,self.B,self.A]
            x0 = self.X[-1]

            if not self._use_constraints:
                obj = EHVI.get_EHVI
            else:
                obj = self._constr_EHVI

            if isinstance(optimizer,optimizers.ParallelLineOpt):
                #get PF x values as x0
                PF_indicies = pareto.get_PF_indicies(self.F)
                x0 = self.X[PF_indicies]

                
                
            res = optimizer.minimize(self.bounds,
                                     obj,
                                     args = fargs,
                                     x0 = x0)
        if return_value:
            return res.x, res.f
        else:
            return res.x

    def plot_acq(self,ax = None):
        if ax is None:
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
        c = ax.pcolor(xx,yy,f)
        ax.figure.colorbar(c,ax=ax)          
            
    def _constr_EHVI(self,x,*args):
        cval = np.array([ele.predict(x.reshape(-1,self.input_dim)) for ele in self.constraints])
        constr_val = np.prod(cval)
        self.constraint_vals = cval
        return EHVI.get_EHVI(x,*args) * constr_val





        
