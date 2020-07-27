import numpy as np
import numdifftools as nd

import advanced_RBF

class HessianRBF(advanced_RBF.AdvancedRBF):
    def __init__(self,variance = 0.1,**kwargs):
        super().__init__(name = 'hessianRBF',**kwargs)

    def update_precision(self,x,f,*args):
        self.dim = x.shape[-1]
        
        #define temp function to handle external args
        def _f_temp(x):
            return f(x,*args)

        hessian =  nd.Hessian(_f_temp)(x)

        self.precision = - hessian / (2 * _f_temp(x))
        print(self.precision)
        
        #decompose the precision matrix into L * L.T
        L = np.linalg.cholesky(self.precision)

        self.S = L[np.triu_indices(self.dim)]

if __name__ == '__main__':
    pass
    #kernel = HessianRBF(S = np.ones(1))
    
    #def f(x,scale):
    #    return scale * np.sin(x)

    #x0 = np.array((0.0)).reshape(-1,1)
    #print(kernel.update_precision(x0,f,1.0))
    #print(kernel.S)
