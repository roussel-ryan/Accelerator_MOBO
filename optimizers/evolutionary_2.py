import numpy as np
import pygmo as pg
import time
import logging


from . import base

class OptProblem:
    def __init__(self, bounds, obj_func, args):
        self.dim = len(bounds)
        self.args = args
        self.bounds = bounds
        
        self.obj_func = obj_func
        
    def fitness(self,x):
        return self.obj_func(x,*self.args)

    def get_bounds(self):
        bounds = tuple(map(tuple,self.bounds.astype(np.float64).T))
        return bounds

    
class PFSwarmOpt(base.BlackBoxOptimizer):
    def __init__(self,**kwargs):
        #swarm optimization hyperparameters
        self.generations = kwargs.get('generations',20)
        self.pop_size    = kwargs.get('population', 50)
        self.islands   = kwargs.get('islands',1)

        self.logger            = logging.getLogger(__name__)

        
    def minimize(self, bounds, func, args = [], x0 = None):
        p = OptProblem(bounds, func, args)
        problem = pg.problem(p)

        algo = pg.algorithm(pg.pso_gen(gen = self.generations))
        #algo.set_verbosity(5)
        pop = pg.population(problem,size = self.pop_size)

        #set population individuals to PF points
        model = args[0]
        Y = model.get_data('Y')
        X = model.get_data('X')

        ndf,dl,dc,ndl = pg.fast_non_dominated_sorting(Y)
        self.logger.info(ndf)
        for ele in ndf:
            np.random.shuffle(ele)
        ndf = np.hstack(ndf)
        #self.logger.info(ndf)
        
        
        self.logger.info('changing population individuals')
        #self.logger.info(pop.get_x())

        if len(ndf) > self.pop_size:
            iters = self.pop_size
        else:
            iters = len(ndf)
            
        for i in range(iters):
            pop.set_x(i, X[ndf[i]])

        #self.logger.info(f'initial population\n {pop.get_x()}')
        
            
        self.logger.info('Starting swarm optimization')
        pop = algo.evolve(pop)
        #isl.wait_check()
        self.logger.info('Done with swarm optimization')

        #self.logger.info(pop.get_x())
        #self.logger.info(pop.get_f())
        #pop = isl.get_population()
        #create result object
        best_x = pop.get_x()[pop.best_idx()]
        best_f = pop.get_f()[pop.best_idx()]
        #self.logger.info(f'final population\n {pop.get_x()}')
        #self.logger.info(f'final population values\n {pop.get_f()}')
        

        self.logger.info(f'best x: {best_x}')
        for gpr in model.GPRs:
            pred = gpr.predict_y(best_x.reshape(1,-1))
            self.logger.info(f'f pred: {pred[0].numpy()} +/- {np.sqrt(pred[1].numpy())}')
        
        self.logger.info(f'best f: {best_f}')
        
        #self.logger.info([gpr.predict_f(best_x.reshape(1,-1)) for gpr in model.GPRs])
        
        res = base.Result(best_x, best_f )
        
        return res

