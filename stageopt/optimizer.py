import numpy as np
import matplotlib.pyplot as plt

from . import confidence
from . import expansion
from . import plotting

import copy
import logging

import time

np.seterr(invalid='ignore')

class StageOpt:
    def __init__(self, inputs):
        self.gprf          = inputs.get('gprf')
        self.gprc          = inputs.get('gprc')
        self.X0            = inputs.get('X0')
        self.Y0            = inputs.get('Y0')
        self.C0            = inputs.get('C0')
        self.ofun          = inputs.get('ofun')
        self.cfun          = inputs.get('cfun')
        self.h             = inputs.get('h')

        self.D             = inputs.get('D',None)
        self.verbose       = inputs.get('verbose',False)
        self.beta          = inputs.get('beta',1.0)
        self.epsilon_max   = inputs.get('epsilon_max',0.1)
        self.T0            = inputs.get('T0',5)
        self.T             = inputs.get('T',20)
        self.n_grid        = inputs.get('n_grid',100)
        self.t             = 1

        
        #initialize domain grid if one does not exist
        if not isinstance(self.D,np.ndarray):            
            self.bounds        = inputs.get('bounds')
            self.input_dim     = len(self.bounds)
            self._initialize_domain()
        else:
            self.input_dim     = len(self.D[0])

                
        self._check_inputs()
        
        #initialize GP regressors
        #measurements
        self.X = [self.X0]
        self.Y = [self.Y0]
        self.C = [self.C0]

        
        self._train_GPs(0)        
        
        self._add_observation_points()    
        self.npts = len(self.D)

        #get mask for initial safe set
        mask = np.ones((self.npts,self.input_dim))
        mask[-len(self.X0):,:] = 0
        S0 = np.ma.masked_array(self.D,mask=mask)
        
        #now sort and initialize sets
        self.D = self.D[self.sort_order.flatten()]
        self.S = [S0[self.sort_order.flatten()]]
        self.G = [None]
        del self.sort_order

        logging.info(self.D)
        self._initialize_confidences()
        
#        self._expand()
#        self._optimize()
        
    def expand(self):
        #expand the safe set using the constraint functions
        logging.info('running expansion')
        start = time.time()
        while self.t < self.T0:
            logging.info(f'running safe expansion step {self.t}')

            #update modified confidences
            confidence.update_modified_confidences(self.t,self)

            #if any of the condititons are < h return True (which masks the array)
            self.S.append(np.ma.masked_where(np.any(self.cond,axis = 0).reshape(-1,1),self.D).flatten())

            #expansion set finding
            expansion.get_expander_points(self.t,self)
            
            #plotting for diagnostic purposes
            if self.verbose:
                fig,ax = plotting.plot_conditions(self.t,self)

            w = self._get_w(self.t)
            epsilon = np.ma.max(w,axis=0)

            exp_done = False
            if np.all(epsilon < self.epsilon_max):
                tempQf = np.ma.masked_where(self.S[self.t].mask,self.Qf[self.t-1].T[1])
                x = self.D[np.ma.argmax(tempQf)]
                logging.info('expander done!')
                exp_done = True
            else:
                x = self.D[np.ma.argmax(np.ma.mean(w,axis=1))]

            if self.verbose:
                for o in range(self.n_cond):
                    ax[o].plot(x,0.0,'o',label='Next point')
                    ax[o].legend()
                    
            #preform measurements
            self.X.append(np.vstack((self.X[self.t-1],x)))
            self.Y.append(np.vstack((self.Y[self.t-1],self.ofun(x))))

            newC = np.empty(self.n_cond)
            for i in range(self.n_cond):
                newC[i] = self.cfun[i](x)
            self.C.append(np.vstack((self.C[self.t-1],newC)))

            #retrain GPs
            self._train_GPs(self.t)
            
            #update confidences
            confidence.update_confidences(self.t,self)

            self.t += 1
            
            if exp_done:
                break

        logging.info(f'expansion done, t: {time.time() - start}')
    def optimize(self):

        while self.t < self.T:
            #optimize the function within the safe set
            logging.info(f'running optimization expansion step {self.t}')

            #update modified confidences
            confidence.update_modified_confidences(self.t,self)

            #if any of the condititons are < h return True (which masks the array)
            self.S.append(np.ma.masked_where(np.any(self.cond,axis = 0).reshape(-1,1),self.D).flatten())

            #optimize using safe points and UCB (Qf[self.t-1].T[1])
            tempQf = np.ma.masked_where(self.S[self.t].mask,self.Qf[self.t-1].T[1])
            x = self.D[np.ma.argmax(tempQf)]

            #preform measurements
            self.X.append(np.vstack((self.X[self.t-1],x)))
            self.Y.append(np.vstack((self.Y[self.t-1],self.ofun(x))))

            logging.info(f'objective function value {self.ofun(x)}')
            
            newC = np.empty(self.n_cond)
            for i in range(self.n_cond):
                newC[i] = self.cfun[i](x)
            self.C.append(np.vstack((self.C[self.t-1],newC)))

                        
            #retrain GPs
            self._train_GPs(self.t)
            
            #update confidences
            confidence.update_confidences(self.t,self)

            #stop if objective function max stays the same within some tolerance (note resolution is given by grid size)
            if np.isclose(self.Y[self.t][-1],self.Y[self.t-1][-1]):
                logging.info('optimization done!')
                break
            
            
            self.t += 1
            
            
        
    def _train_GPs(self,t):
        self.gprf.fit(self.X[t],self.Y[t])

        for i in range(self.n_cond):
            self.gprc[i].fit(self.X[t],self.C[t][:,i])

    def _get_confidence(self,x,gpr):
        mu, std = gpr.predict(x,return_std=True)
        return np.array((mu.flatten() - self.beta*std,mu.flatten() + self.beta*std)).T
            
    def _get_w(self,t):
        #if there are no expander points then epsilon is automatically zero
        if len(self.G[t][~self.G[t].mask]) == 0:
            w = np.zeros((self.npts,self.n_cond))
        else:
            w = np.ma.empty((self.npts,self.n_cond))
            for k in range(self.n_cond):
                w[:,k] = self.Ci[t][:,k].T[1] - self.Ci[t][:,k].T[0]
                w[:,k] = np.ma.masked_where(self.G[t].mask,w[:,k])
            #logging.info(w)

        return w

    def _initialize_domain(self):
        g = []
        for i in range(self.input_dim):
            g.append(np.linspace(*self.bounds[i],self.n_grid))
        mesh = np.meshgrid(*g)
        self.D = np.vstack([ele.ravel() for ele in mesh]).T
        
    def _add_observation_points(self):
        #add on observation points
        self.D = np.vstack((self.D,self.X[0]))
        self.sort_order = np.argsort(self.D.T[0],axis=0)

        
    def _check_inputs(self):
        assert self.T > self.T0
        
        if not isinstance(self.gprc,list):
            self.gprc = [self.gprc]

        if not isinstance(self.cfun,list):
            self.cfun = [self.cfun]

        #number of constraints
        self.n_cond            = len(self.gprc)
        assert self.n_cond == len(self.cfun)
        self.h             = np.atleast_1d(self.h)
        assert self.n_cond == len(self.h)
        assert self.C0.shape[1] == self.n_cond
        
        #check + modify inputs if necessary
        for c in self.C0:
            assert np.all(c > self.h)

    def _initialize_confidences(self):
        #initialize modified confidences
        C0i = np.empty(shape = (self.npts,self.n_cond,2))
        C0i[:,:,0] = -np.inf
        C0i[:,:,1] = np.inf

        C0f = np.empty(shape = (self.npts,2))
        C0f[:,0] = -np.inf
        C0f[:,1] = np.inf
        

        #where measurements occur
        logging.info(self.S[0].mask)
        for i in range(self.n_cond):
            C0i[~self.S[0].mask.T[0],i,0] = self.h[i]

        #initialize confidences
        Qf0 = self._get_confidence(self.D,self.gprf)
        q = np.empty((self.n_cond,self.npts,2))
        for i in range(self.n_cond):
            q[i] = self._get_confidence(self.D,self.gprc[i])
        Qi0 = np.transpose(q,(1,0,2))
        
        #initialize storage vectors        
        #confidence intervals
        self.Qf = [Qf0]
        self.Qi = [Qi0]

        #modified confidence intervals
        self.Ci = [C0i]
        self.Cf = [C0f]
