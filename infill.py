import numpy as np
from scipy.stats import multivariate_normal
from .multi_objective import uhvi
from .multi_objective import biuhvi
import time
import logging

class Infill:
    def __init__(self,name):
        self.name = name


class Restricted_UHVI(Infill):
    def __init__(self, cov = 0.25, **kwargs):
        self.cov = cov
        self.uhvi = UHVI(**kwargs)
        print(self.uhvi.beta)
        
    def __call__(self, X, model):
        #get last point
        GPRs = model.GPRs
        x0 = GPRs[0].data[0][-1]
        alpha0 = self.uhvi(X, model)
        
        return alpha0 * multivariate_normal.pdf(X, mean = x0, cov = self.cov)
        #return alpha0
        #if np.linalg.norm(X - x0) < 1.5:
            #print(self.uhvi(X, GPRs, PF, A, B))
        #    return self.uhvi(X, GPRs, PF, A, B)
        #else:
        #    return np.array([0.0])

        
class UHVI(Infill):
    def __init__(self, beta = None, D = None, delta = None, approx = False, use_bi = False):
        self.D       = D
        self.delta   = delta
        self.approx  = approx
        self.t       = 1
        self.use_bi  = use_bi 
        self.eval_time = []
        
        super().__init__('uhvi')
        
        #if beta is not specified use the beta schedule
        if beta == None:
            self.use_schedule = True
            
            assert not self.D == None
            assert not self.delta == None

        else:
            self.use_schedule = False
            self.beta = beta

    def __call__(self, X, model):
        GPRs = model.GPRs
        PF = model.get_PF()
        A = model.A
        B = model.B
        
        start = time.time()
        if self.approx:
            res = uhvi.get_approx_uhvi(X, GPRs, PF, A, B, self.get_beta())
        else:
            #if self.use_bi:
            #    res = biuhvi.get_biuhvi(X, GPRs, PF, A, B, self.get_beta())
            #else:
            res = uhvi.get_uhvi(X, GPRs, PF, A, B, self.get_beta())
        self.eval_time += [time.time() - start]
        return res

    def get_avg_time(self):
        return np.mean(np.array(self.eval_time))

    def reset_timer(self):
        self.eval_time = []
    
    def get_beta(self):
        if self.use_schedule:
            return 2 * np.log(self.D * self.t**2 * np.pi**2 / (6 * self.delta))
        else:
            return self.beta


class UCB(Infill):
    def __init__(self, beta = 0.1):

        self.beta    = beta
        self.t       = 1

        super().__init__('ucb')
        
    def __call__(self, X, GPR):
        p = GPR.predict_y(np.atleast_2d(X))
        return p[0].numpy() + np.sqrt(self.beta * p[1].numpy())
    
    
