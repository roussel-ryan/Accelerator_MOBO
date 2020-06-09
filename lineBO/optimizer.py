import numpy as np
import matplotlib.pyplot as plt
import logging

import scipy.optimize as opt
from . import oracles

class LineOpt:
    def __init__(self,bounds,acq,**kwargs):
        '''
        Uses line optimiztion to solve a high dimentional global optimization problem
        
        Arguments
        ---------
        bounds                np array containing ((x_1_min,x_1_max),(x_2_min,x_2_max),...)
        acq                   function of the form f(x,*args) that is to be minimized
        
        Optional Arguments
        ------------------
        args                  arguments for optimization function (default: [])
        oracle                direction choosing oracle (default: random)
        X0                    initial point to start optimization (default: np.random.uniform)
        verbose               display diagnostic plotting
        T                     step budget (default:10)
        tol                   convergence tolerance (default: 1e-6)

        '''


        self.dim        = len(bounds)
        self.bounds     = bounds
        self.acq        = acq
        
        #set up oracle and acq_args
        self.oracle = kwargs.get('oracle',oracles.random)
        self.acq_args = kwargs.get('args',[])

        #initialize storage vectors
        self.x = kwargs.get('X0',np.random.uniform(bounds[:,0],bounds[:,1]).reshape(-1,self.dim))
        self.l = np.atleast_2d(self.oracle(self.dim))

        #logging.info((self.x,self.l))
        
        self.verbose = kwargs.get('verbose',False)
        self.T = kwargs.get('T',40)
        self.tol = kwargs.get('tol',1e-6)

        #tracking stats
        self.t = 0
        self.f = []
        self.lower = []
        self.upper = []
        
    
    def _map_subdomain(self,s):
        #maps points in subdomain <s> to real space
        return self.l[self.t]*s + self.x[self.t]
    
    def optimize(self):        
        while self.t < self.T:
            logging.info(f'doing optimization step {self.t}')
            #generate direction from oracle
            self.l = np.vstack((self.l,self.oracle(self.dim)))

            #find subspace bounds
            self._get_subdomain_bounds()

            #get next point via 1D optimization
            self._get_next_point()

            
            if not self.t == 0:
                dist = np.linalg.norm(self.x[self.t] - self.x[self.t-1])
                logging.info(f'distance: {dist}')
                if dist < self.tol:
                    logging.info('optimization done!')
                    break

            self.t += 1
        
        return None
        
    
    def _get_next_point(self):
        ''' find the next point via minimization of acquisition function in subdomain'''
        
        #optimization step
        sub_bounds = np.atleast_2d(np.array((self.lower[self.t],self.upper[self.t])))

        #t_next, func_val = grid_minimization(self._transformed_acq,sub_bounds,(direction[0],x0))
        s_next, func_val = brent_minimization(self._transformed_acq,sub_bounds)

        
        if self.t == 0:
            self.f.append(func_val)
            self.x = np.vstack((self.x,self._map_subdomain(s_next)))
        else:
            if self.f[self.t - 1] < func_val:
                self.x = np.vstack((self.x,self.x[self.t-1]))
                self.f.append(self.f[self.t-1])
            else:
                self.x = np.vstack((self.x,self._map_subdomain(s_next)))
                self.f.append(func_val)
        
        
        if self.verbose:
            self._do_plotting()
            
        
    def _transformed_acq(self,s):
        '''wrapper function for acq that transforms from subdomain var t to real domain var x'''
        x = self._map_subdomain(s)
        #self.n_calls = self.n_calls + 1
        return self.acq(x,*self.acq_args)
    
    def _get_subdomain_bounds(self):
        '''get subdomain bounds'''
        #logging.info('getting subdomain')
        #logging.info(f'direction vector:{direction}')
        #logging.info(f'point:{x0}')
        #make sure these are numpy arrays
        old_lower = np.array(self.bounds.T[0])
        old_upper = np.array(self.bounds.T[1])

        #define the output arrays
        lower = np.empty((1))
        upper = np.empty((1))

        v = self.l[self.t]
        if len(old_lower) != len(v) or len(old_upper) != len(v):
            raise ValueError("Basis needs to have the same dimension than the bounds")
        temp_l = np.empty_like(v)
        temp_u = np.empty_like(v)
        for i in range(len(v)):
            if v[i] > 0:
                temp_u[i] = (old_upper[i]-self.x[self.t][i])/v[i]
                temp_l[i] = (old_lower[i]-self.x[self.t][i])/v[i]
            elif v[i] < 0:
                temp_l[i] = (old_upper[i]-self.x[self.t][i])/v[i]
                temp_u[i] = (old_lower[i]-self.x[self.t][i])/v[i]
            else:
                temp_l[i] = -np.inf
                temp_u[i] = np.inf
        #we use the minimum distance to the boundaries to define our new bounds
        self.upper.append(np.min(temp_u))
        self.lower.append(np.max(temp_l))

        #return lower, upper

    def _do_plotting(self):
        fig,(ax,ax2) = plt.subplots(2,1)
        n = 30
        x = np.linspace(*self.bounds[0],n)
        y = np.linspace(*self.bounds[1],n)
        xx,yy = np.meshgrid(x,y)
        pts = np.vstack((xx.ravel(),yy.ravel())).T
        f = np.array([self.acq(pt,*self.acq_args) for pt in pts])
        #plot function
        ax.pcolor(xx,yy,f.reshape(n,n))

        #plot subspace line
        s = np.linspace(self.lower[self.t],self.upper[self.t])
        sub = np.array([self._map_subdomain(ele) for ele in s])
        ax.plot(*sub.T,'r+')
        ax.plot(*self.x[self.t - 1],'o')
        ax.plot(*self.x[self.t],'ro')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
            
        sub_f = np.array([self.acq(ele,*self.acq_args) for ele in sub])
        ax2.plot(s,sub_f)
        #ax2.axvline(t_next,color='r')
        ax2.axvline(0)
        ax2.set_ylabel('$f$')
        ax2.set_xlabel('t')
        

def brent_minimization(func,sbounds):
    res = opt.minimize_scalar(func,bounds=sbounds[0],method='Bounded')
    return res.x, res.fun
    
def grid_minimization(func,bounds,args):
    n = 10
    pts = np.linspace(*bounds[0],n)
    _min = 100000000
    for pt in pts:
        res = opt.minimize(func,np.atleast_1d(pt),args=args,bounds=bounds,options={'maxiter':10})
        if res.fun < _min:
            _min = res.fun
            _minx = res.x
    return _minx, _min
