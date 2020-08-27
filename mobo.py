import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import time

import pygmo as pg
import tensorflow as tf

from .multi_objective import EHVI
from .multi_objective import pareto

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
        if not self._use_constraints:
            self.logger.info('Using constraint function')
            self.obj = self.infill
        else:
            self.obj = self.constrained_infill

            
        #create a pandas dataframe to store info about observations
        self.update_model_data()

        self.t = 0
        self.history = []

    def update_model_data(self):
        #add data from GPRs
        x_data = self.GPRs[0].data[0].numpy()

        #store the number of observed points in the model
        self.n_observations = len(x_data)

        y_data = np.hstack([ele.data[1].numpy() for ele in self.GPRs])
        
        
        frame_cols = {}
        for i in range(self.input_dim):
            frame_cols[f'X{i}'] = x_data.T[i]

        for j in range(self.obj_dim):
            frame_cols[f'Y{j}'] = y_data.T[j]

        if self._use_constraints:
            c_data = np.hstack([ele.GPR.data[1].numpy() for ele in self.constraints])
            
            for k in range(self.constr_dim):
                frame_cols[f'C{k}'] = c_data.T[k]

            frame_cols['is_feasable'] = self.get_feasable_labels().astype(bool).tolist()    

        frame_cols['in_target_range'] = self.inside_obj_domain(y_data)
                                   
            
        self.data = pd.DataFrame(frame_cols)

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
        #do the same for C
        npts = Y.shape[0]
        Y = Y.reshape(self.obj_dim, npts, 1)
        C = C.reshape(self.constr_dim, npts, 1)
        
        for i in range(self.obj_dim):
            #add observed data to GPRs
            y_data = Y[i]
            gpr = self.GPRs[i]
            gpr.data = (tf.concat((gpr.data[0],X),axis=0),
                        tf.concat((gpr.data[1],y_data),axis=0))

        for j in range(self.constr_dim):
            self.constraints[j].add_observations(X,C[j])

        self.update_model_data()
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
        self.logger.info(f'UHVI beta {self.infill.get_beta()}')

        res = optimizer_func(self.bounds, _neg_obj, args)

        exec_time = time.time() - start
        self.logger.info(f'Done with optimization in {exec_time} s')

        stats = pd.DataFrame({'exec_time':exec_time,
                              'n_obs':self.n_observations,
                              'n_iterations' : self.t,
                              'n_pf':len(self.PF),
                              'hypervolume':self.get_hypervolume(),
                              'predicted_ideal_point':[res.x],
                              'predicted_hypervolume_improvment':np.abs(res.f),
                              'actual_hypervolume_improvement':np.nan,
                              'log_marginal_likelihood':[self.get_log_marginal_likelihood()]},
                             index = [self.t])

        if isinstance(self.infill,infill.UHVI):
            stats['beta'] = self.infill.get_beta()
        
        if isinstance(self.history,pd.DataFrame):
            self.history.at[self.t-1,'actual_hypervolume_improvement'] = stats['hypervolume'] - self.history.at[self.t-1,'hypervolume'] 
            self.history = pd.concat([self.history,stats])
        else:
            self.history = stats

        self.t += 1

        return res

    def get_feasable(self,invert = False):
        if invert:
            return self.data[~(self.data['is_feasable'] & self.data['in_target_range'])]
        else:
            return self.data[self.data['is_feasable'] & self.data['in_target_range']]
        
    def get_feasable_Y(self, invert = False):
        return self.get_feasable(invert).filter(regex='^Y',axis=1).to_numpy()

    def get_feasable_X(self, invert = False):
        return self.get_feasable(invert).filter(regex='^X',axis=1).to_numpy()
    

    
    def get_PF(self):
        F = self.get_feasable_Y()
        return pareto.get_PF(F, self.B, tol = 1e-5)
        
    def get_hypervolume(self):
        hv = pg.hypervolume(self.get_PF())
        return hv.compute(self.B)

    def get_log_marginal_likelihood(self):
        res = np.array([ele.log_marginal_likelihood().numpy() for ele in self.GPRs])
        return res

    def get_feasable_labels(self):
        if self._use_constraints:
            b = []
            for const in self.constraints:
                b += [const.get_feasable()]

            b = np.array(b)
            b = np.prod(b,axis=0)
        else:
            b = np.ones(self.n_observations)
            
        return b
        
    def get_feasable_idx(self):
        return np.argwhere(self.get_feasable_labels()).flatten()

    def inside_obj_domain(self,F):
        return (np.all(F > self.A) and np.all(F < self.B))

    
    def plot_acq(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()

        self.PF = self.get_PF()
        fargs = [self.GPRs,self.PF,self.A,self.B]    

        n = 30
        x = np.linspace(*self.bounds[0,:],n)
        y = np.linspace(*self.bounds[1,:],n)
        xx, yy = np.meshgrid(x,y)
        pts = np.vstack((xx.ravel(),yy.ravel())).T

        f = []
        for pt in pts:
            f += [self.obj(pt,*fargs)]

        f = np.array(f).reshape(n,n)
        
        c = ax.pcolor(xx,yy,f)
        ax.figure.colorbar(c,ax=ax)          

        ax.plot(*self.GPRs[0].data[0].numpy().T,'+')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        
        return ax

    def plot_constr(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        
        fargs = [self.GPRs,self.PF,self.A,self.B]    

        n = 30
        x = np.linspace(*self.bounds[0,:],n)
        y = np.linspace(*self.bounds[1,:],n)
        xx, yy = np.meshgrid(x,y)
        pts = np.vstack((xx.ravel(),yy.ravel())).T

        f = []
        for pt in pts:
            f += [self.constraints[0].predict(np.atleast_2d(pt))]

        f = np.array(f).reshape(n,n)
        
        c = ax.pcolor(xx,yy,f)
        ax.figure.colorbar(c,ax=ax)          

        X_feas = self.get_feasable_X()
        X_nonfeas = self.get_feasable_X(invert = True)
        
        ax.plot(*X_feas.T,'+r')
        ax.plot(*X_nonfeas.T, 'or')

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        
        return ax

        
            
    def constrained_infill(self, x, *args):
        cval = np.array([ele.predict(
            x.reshape(-1,self.input_dim)) for ele in self.constraints])
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


    

    '''    
    def update_objective_data(self):
        data = []
        for gpr in self.GPRs:
            data.append(gpr.data[1])

        self.F = np.array(data).T[0]

        #make sure that any points that are outside the n-D domain [A,B] are excluded
        #from the calculated PF
        self.temp_F = []
        for i in range(len(self.F)):
            in_obj_domain = (np.all(self.F[i] > self.A) and np.all(self.F[i] < self.B))
            #in_obj_domain = (np.all(self.F[i] < self.B))
            
            if not in_obj_domain:
                warn_string = f'Point {self.F[i]} lies outside objective domain, '
                warn_string += 'it has been taken out of PF calculations but it still remains in the training set'
                self.logger.warning(warn_string)
            else:
                self.temp_F += [self.F[i]]
        self.F = np.vstack(self.temp_F)

        logging.debug(self.F)
        
        #make sure that we only use points that satisfy the constraint(s)
        if self._use_constraints:
            f_idx = self.get_feasable_idx()
            self.F = self.F[f_idx]
            
        logging.debug(self.F)
        self.PF = self.get_PF()
    ''' 
        
