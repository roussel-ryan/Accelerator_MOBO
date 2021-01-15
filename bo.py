#base BayesianOptimizer class

import numpy as np
import pandas as pd

import logging
import time
import pickle

import tensorflow as tf
import gpflow

from . import utilities

class BayesianOptimizer:
    """Bayesian optimizer class

    This class serves as the base class for single and multiple objective bayesian 
    optimizer classes.

    Attributes:
    -----------
    bounds : sequence
        Sequence of (min,max) pairs that defines design/input space

    optimizer : Optimizer object
        Optimizer object used to optimize the acquisition function

    acq : callable
        Callable function that calculates the acquisition function \alpha(x)
        must be in the form of f(x,model)

    """
    def __init__(self, bounds, optimizer, acq):
        self.bounds = bounds
        self.optimizer = optimizer
        self.acq = acq

        self.domain_dim = len(bounds)
        self.obj_dim = self.get_obj_dim()
        self.constr_dim = self.get_constr_dim()
        
        self.data = None
        self.stats = None

        self.logger = logging.getLogger(__name__)
        


    def get_next_point(self):
        #call acquisition function pre-optimizer
        self.acq.pre_opt(self)
        
        def _neg_obj(x, model):
            return -1.0 * self.acq(np.atleast_2d(x), model)

        start = time.time()
        self.logger.info('Starting acquisition function optimization')
        res = self.optimizer.minimize(self.bounds, _neg_obj, self)
        exec_time = time.time() - start
        self.logger.info(f'Done with optimization in {exec_time} s')

        opt_stats = {'exec_time':exec_time, **self._get_optimization_stats()}
        self.logger.info(f'Avg. exec time : {self.acq.get_avg_time()} s')
        
        df = pd.DataFrame.from_dict(opt_stats,orient='index')
        if isinstance(self.stats,pd.DataFrame):
            self.stats = pd.concat((self.stats,df))
        else:
            self.stats = df

        return res

    def add_observations(self, X, Y, Z = {}, reopt = False):
        self._add_to_dataframe(X, Y, Z)
        self._add_to_GP(X, Y, Z)
    
    def _add_to_dataframe(self, X, Y, Z = {}):
        #adds data observations to self.data pd.DataFrame object
        # z is a dict of corresponding values that can be added at runtime
        # for example could be constraints {'C1': [C11, C12, ...]}
        
        assert X.shape[1] == self.domain_dim
        assert Y.shape[1] == self.obj_dim
        assert X.shape[0] == Y.shape[0]

        #if dataframe has been created, import the data to append to
        cols = [f'X{i}' for i in range(self.domain_dim)] + \
            [f'Y{i}' for i in range(self.obj_dim)]

        data_array = np.hstack((X,Y))
        for key, item in Z.items():
            cols += key
            data_array = np.hstack((data_array,item))
            
        col_mapping = {}
        for i in range(len(cols)):
            col_mapping[i] = cols[i]

        
        df = pd.DataFrame(data_array)    
        df = df.rename(columns = col_mapping)

        
        if isinstance(self.data,pd.DataFrame):
            self.data = pd.concat([self.data, df], ignore_index = True)
        else:
            self.data = df

    def _clear_dataframe(self):
        self.data = None

    def _train_hyp(self, gpr, max_iter = 5000, lr = 0.01):
        self.logger.info('training hyperparameters')
        opt = tf.optimizers.Adam(learning_rate = lr)

        old_lml = -1e9
        lml_deltas = []
        
        for i in range(max_iter):
            if i % 100 == 0:
                lml = gpr.log_marginal_likelihood().numpy()
                delta = np.abs((old_lml - lml) / lml)
                lml_deltas += [delta]
                self.logger.info(f'{i}:{lml},{delta}')

                avg_lml_delta = np.mean(np.array(lml_deltas)[-5:])
                if avg_lml_delta < 0.01:
                    break
                else:
                    old_lml = lml
            opt.minimize(gpr.training_loss, gpr.trainable_variables)

    def _print_model(self,model):
        gpflow.utilities.print_summary(model)
            
    def get_data(self, name = 'all', valid = None, convert = True):
        return utilities.get_data(self, name, valid, convert)

    def save(self, fname):
        pickle.dump(self, open(fname,'wb'))

    def _add_to_GP(self,X, Y, Z):
        raise NotImplementedError

    def _get_optimization_stats(self):
        return {}
            
    def get_obj_dim(self):
        raise NotImplementedError

    def get_constr_dim(self):
        return 0

    

    
class TDOptimizer:
    def __init__(self,time = 0):
        self.time = time

    
class Test(BayesianOptimizer):
    def __init__(self):
        super().__init__([1,2],None,None)

    def get_obj_dim(self):
        return 1


    
if __name__=='__main__':
    b = Test()
    x = np.ones((2,2))
    y = np.zeros((2,1))
    b.add_observations(x,y)
    b.add_observations(x,y)
