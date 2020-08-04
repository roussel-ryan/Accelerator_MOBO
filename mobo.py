import numpy as np
import matplotlib.pyplot as plt

import logging
import time

import pygmo as pg
import tensorflow as tf

from .multi_objective import EHVI
from .multi_objective import UHVI

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

        self.logger            = logging.getLogger(__name__)

        self.history           = []

        #set infill function
        infill                 = kwargs.get('infill','UHVI')
        if infill == 'UHVI':
            self.infill = UHVI.get_UHVI

        elif infill == 'EHVI':
            self.infill = EHVI.get_EHVI

        else:
            raise RuntimeError(f'infill function {self.infill} not found!')

        #add constraints if necessary
        if not self._use_constraints:
            self.obj = self.infill
        else:
            self.obj = self.constrained_infill
        
        self.update_objective_data()
        
    def update_objective_data(self):
        data = []
        for gpr in self.GPRs:
            data.append(gpr.data[1])

        self.F = np.array(data).T[0]

        #make sure that any points that are outside the n-D domain [A,B] are excluded
        #from the calculated PF
        self.temp_F = []
        for i in range(len(self.F)):
            #in_obj_domain = (np.all(self.F[i] > self.A) and np.all(self.F[i] < self.B))
            in_obj_domain = (np.all(self.F[i] < self.B))
            
            if not in_obj_domain:
                warn_string = f'Point {self.F[i]} lies outside objective domain, '
                warn_string += 'it has been taken out of PF calculations but it still remains in the training set'
                self.logger.warning(warn_string)
            else:
                self.temp_F += [self.F[i]]
        self.F = np.vstack(self.temp_F)
        
        self.PF = self.get_PF()
        
        
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
        npts = Y.shape[0]
        Y = Y.reshape(self.obj_dim, npts, 1)
        
        for i in range(self.obj_dim):
            #add observed data to GPRs
            y_data = Y[i]
            gpr = self.GPRs[i]
            gpr.data = (tf.concat((gpr.data[0],X),axis=0),
                        tf.concat((gpr.data[1],y_data),axis=0))

        self.update_objective_data()
        #TODO: add constraint data acceptance
        
    def get_next_point(self, optimizer_func, return_value = False, **kwargs):
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
        res = optimizer_func(self.bounds, _neg_obj, args, **kwargs)
        self.logger.info(f'Done with optimization in {time.time() - start} s')

        return res

    def get_PF(self):
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(self.F)
        return self.F[ndf[0]]
    
    def get_hypervolume(self):
        hv = pg.hypervolume(self.PF)
        return hv.compute(self.B)

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

        
            
    def constrained_infill(self, x, *args):
        cval = np.array([ele.predict(x.reshape(-1,self.input_dim)) for ele in self.constraints])
        constr_val = np.prod(cval)
        self.constraint_vals = cval
        return self.infill(x,*args) * constr_val

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


    

        
