import numpy as np
from scipy.stats import multivariate_normal
from .multi_objective import uhvi
from .multi_objective import biuhvi
import time
import logging

class Infill:
    def __init__(self,name, **kwargs):
        self.name = name
        self.settings = kwargs
        self.eval_time = []
        self.n_calls = 0
        self.acq_func = None
        
        
    def __call__(self,X,model):
        start = time.time()
        res = self.acq_func(X, model)
        self.eval_time += [time.time() - start]
        return res
        

    def get_avg_time(self):
        return np.mean(np.array(self.eval_time))

    def reset(self):
        self.eval_time = []
        self.n_calls = 0

    def add_defaults(self,d):
        for key in d:
            if not key in self.settings:
                self.settings[key] = d[key]
        
class UHVI(Infill):
    def __init__(self, name = 'UHVI', **kwargs):
        #add default values
        d = {'beta':2.0, 'use_approx':False, 'use_schedule': False,
             'D':0.0, 'delta': 1.0}

        super().__init__(name, **kwargs)
        self.add_defaults(d)
        self.acq_func = self.get_UHVI
        
    def get_UHVI(self, X, model):
        GPRs = model.GPRs
        PF = model.get_PF()
        A = model.A
        B = model.B
        
        if self.settings.get('use_approx',False):
            res = uhvi.get_approx_uhvi(X, GPRs, PF, A, B, self.get_beta())
        else:
            res = uhvi.get_uhvi(X, GPRs, PF, A, B, self.get_beta())
        
        return res

    
    def get_beta(self):
        D = self.settings['D']
        delta = self.settings['delta']
        use_schedule = self.settings['use_schedule']

        if use_schedule:
            return 2 * np.log(D * self.n_calls**2 * np.pi**2 / (6 * delta))
        else:
            return self.settings['beta']

        
class SUHVI(UHVI):
    def __init__(self, **kwargs):
        d = {'cov':0.25}
        
        super().__init__('SUHVI', **kwargs)
        self.add_defaults(d)
        self.acq_func = self.get_SUHVI
        
    def get_SUHVI(self, X, model):
        #get last point
        GPRs = model.GPRs
        x0 = GPRs[0].data[0][-1]
        alpha0 = self.get_UHVI(X, model)
        
        return alpha0 * multivariate_normal.pdf(X, mean = x0, cov = self.settings['cov'])
       
