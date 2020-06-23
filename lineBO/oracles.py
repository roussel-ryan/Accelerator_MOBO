import numpy as np

import logging

class Oracle:
    def __init__(self,name):
        self.name = name

        
class RandomOracle(Oracle):
    def __init__(self,dim):
        super().__init__('RandomOracle')
        self.dim = dim
        
    def get_direction(self,x0,acq,*args):
        ''' returns random unit vector to determine direction'''
        d = np.random.uniform(-1,1,size = self.dim)
        return d / np.linalg.norm(d)

    
class GradientOracle(Oracle):
    def __init__(self,bounds,delta = 0.01):
        super().__init__('GradientOracle')
        self.bounds = bounds
        self.dim = len(bounds)
        self.widths = self.bounds[:,1] - self.bounds[:,0]
        self.delta = delta
        
    def get_direction(self,x0,acq,*args):
        '''calcuate the direction by using the gradient \
        (requires extra calls to acq)
    
        '''
        grad = np.zeros(self.dim)
        f0 = acq(x0,*args)
        for i in range(self.dim):
            dx = np.zeros(self.dim)
            dx[i] = 1.0 * self.widths[i] * self.delta 
            grad[i] = (acq(x0 + dx,*args) - f0) / dx[i]
            
            
        #return normalized vector direction if gradient is nonzero
        #otherwise return a random direction,
        #if we are at a local minima we will stay there
        if not np.linalg.norm(grad) == 0.0:
            return -grad / np.linalg.norm(grad)
        else:
            d = np.random.uniform(-1,1,size = self.dim)
            return d / np.linalg.norm(d)
            
            
            


if __name__ =='__main__':
    def f(x):
        return np.linalg.norm(x)

    print(1)
    bounds = np.array(((-1,1),(-1,1)))
    g = GradientOracle(bounds)
    x0 = np.array((-0.5,0.5))
    print(g.get_direction(x0,f))
