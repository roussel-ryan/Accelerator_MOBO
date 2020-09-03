import numpy as np
from .multi_objective import uhvi

import time

class Infill:
    def __init__(self,name):
        self.name = name

class UHVI(Infill):
    def __init__(self, beta = None, D = None, delta = None, approx = False):
        self.D       = D
        self.delta   = delta
        self.approx  = approx
        self.t       = 1

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

    def __call__(self, X, GPRs, PF, A, B):
        start = time.time()
        if self.approx:
            res = uhvi.get_approx_uhvi(X, GPRs, PF, A, B, self.get_beta())
        else:
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
    
    
