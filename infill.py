import numpy as np
from scipy.stats import multivariate_normal
from .multi_objective import uhvi
from .multi_objective import biuhvi
from .multi_objective import pareto
from .multi_objective import td_pareto
import time
import logging

from . import bo

class Infill:
    def __init__(self,name, **kwargs):
        self.name = name
        self.settings = kwargs
        self.eval_time = []
        self.n_calls = 0
        self.acq_func = None

        self.logger = logging.getLogger(__name__)
        
    def __call__(self, X, model):
        assert len(X.shape) == 2
        start = time.time()
        res = self.acq_func(X, model)
        self.eval_time += [time.time() - start]
        return res

    def pre_opt(self,model):
        self.reset()

    def get_avg_time(self):
        return np.mean(np.array(self.eval_time))

    def reset(self):
        self.eval_time = []
        self.n_calls = 0

    def add_defaults(self,d):
        for key in d:
            if not key in self.settings:
                self.settings[key] = d[key]

class TDACQ(Infill):
    def __init__(self, base, **kwargs):
        self.base_acq = base
        super().__init__('TD' + base.name)
        self.add_defaults(base.settings)
        self.acq_func = self.get_tdacq

    def pre_opt(self, model):
        self.base_acq.pre_opt(model)
        
    def get_tdacq(self, X, model):
        assert isinstance(model,bo.TDOptimizer)
        #append model time to input X
        T = np.ones(len(X)).reshape(-1,1) * model.time
        X = np.hstack((X,T))
        return self.base_acq.acq_func(X, model)
                
class UCB(Infill):
    def __init__(self, name = 'UCB', **kwargs):
        d = {'beta':2.0, 'maximize' : True}
        super().__init__(name, **kwargs)
        self.add_defaults(d)
        self.acq_func = self.get_UCB

    
    def get_UCB(self, X, model):
        gpr = model.GPR
        p = gpr.predict_y(X)
        m = p[0]
        s = p[1]

        if self.settings['maximize']:
            val = m + np.sqrt(self.settings['beta'] * s)
        else:
            val = -1 * (m - np.sqrt(self.settings['beta'] * s))
            
        return val.numpy().flatten()[0]
        
class UHVI(Infill):
    def __init__(self, name = 'UHVI', **kwargs):
        #add default values
        d = {'beta':2.0, 'use_approx':False, 'use_schedule': False,
             'D':0.0, 'delta': 1.0, 'use_bidirectional': False}

        super().__init__(name, **kwargs)
        self.add_defaults(d)
        self.acq_func = self.get_UHVI

    def pre_opt(self, model):
        self.PF = model.get_PF()
        self.logger.debug(f'PF : {self.PF}')

    def _get_PF(self):
        return self.PF
        
    def get_UHVI(self, X, model):
        GPRs = model.GPRs
        PF = self._get_PF()
        A = model.A
        B = model.B

        self.logger.debug(f'X : {X}')
        
        F = uhvi.get_predicted_uhvi_point(X, GPRs, self.get_beta())
        res = uhvi.get_HVI(F, PF, A, B,
                           use_bi = self.settings['use_bidirectional'],
                           use_approx = self.settings['use_approx'])

        self.logger.debug(f'result : {res}')
        return res

    
    def get_beta(self):
        D = self.settings['D']
        delta = self.settings['delta']
        use_schedule = self.settings['use_schedule']

        if use_schedule:
            return 2 * np.log(D * self.n_calls**2 * np.pi**2 / (6 * delta))
        else:
            return self.settings['beta']

class NUHVI(UHVI):
    def __init__(self, **kwargs):
        d = {'gamma':1.0}
        super().__init__('NUHVI', **kwargs)
        self.add_defaults(d)
        self.acq_func = self.get_UHVI

    def _get_PF(self):
        return self.PCB_PF
        
    def pre_opt(self, model):
        self.reset()
        self.logger.info('calculating PCB PF')
        self.PCB_PF = td_pareto.get_PCB_PF(model, gamma = self.settings['gamma'])

        
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
       
class Restricted_UHVI(Infill):
    def __init__(self):
        pass
