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

class CMAESOpt(base.BlackBoxOptimizer):
    def __init__(self,**kwargs):
        #swarm optimization hyperparameters
        self.generations = kwargs.pop('generations',20)
        self.pop_size    = kwargs.pop('population', 50)
        self.islands   = kwargs.pop('islands',1)

        #use remaining kwargs for algorithm keywords
        self.algo_kwargs = kwargs
        
        self.logger            = logging.getLogger(__name__)

        
    def minimize(self, bounds, func, args = [], x0 = None):
        p = OptProblem(bounds, func, args)
        problem = pg.problem(p)

        algo = pg.algorithm(pg.cmaes(gen = self.generations,
                                     **self.algo_kwargs))
       
        pop = pg.population(problem,size = self.pop_size)
        
        self.logger.info('Starting swarm optimization')
        pop = algo.evolve(pop)
        self.logger.info('Done with swarm optimization')

        #create result object
        res = base.Result(pop.get_x()[pop.best_idx()], pop.get_f()[pop.best_idx()])
        
        return res

    
class SwarmOpt(base.BlackBoxOptimizer):
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

        #isl = pg.island(algo = algo,
        #                pop = pop,
        #                udi = pg.mp_island(True))
        self.logger.info('Starting swarm optimization')
        pop = algo.evolve(pop)
        #isl.wait_check()
        self.logger.info('Done with swarm optimization')

        
        #pop = isl.get_population()
        #create result object
        res = base.Result(pop.get_x()[pop.best_idx()], pop.get_f()[pop.best_idx()])
        
        return res

class ParallelSwarmOpt(base.BlackBoxOptimizer):
    def __init__(self,**kwargs):
        #swarm optimization hyperparameters
        self.generations = kwargs.get('generations',200)
        self.pop_size    = kwargs.get('pop_size', 50)
        self.islands   = kwargs.get('islands',5)

        self.logger            = logging.getLogger(__name__)

        
    def minimize(self, bounds, func, args = [], x0 = None):
        p = OptProblem(bounds, func, args)
        problem = pg.problem(p)

        algo = pg.algorithm(pg.pso(gen = self.generations))
        archi = pg.archipelago(n = self.islands,
                               algo = algo,
                               prob = problem,
                               pop_size = self.pop_size)

        #pop = pg.population(problem,size = self.pop_size)
        
        self.logger.info('Starting swarm optimization')
        #self.logger.info(archi)
        archi.evolve()
        archi.wait()
        self.logger.info(archi)
        self.logger.info('Done with swarm optimization')

        champs_f = np.array(archi.get_champions_f()).flatten()
        best_champ_idx = np.argmin(champs_f)
        best_champ_x = archi.get_champions_x()[best_champ_idx]
        best_champ_f = archi.get_champions_f()[best_champ_idx]
        self.logger.info(champs_f)
        
        #create result object
        res = base.Result(best_champ_x,best_champ_f)
        
        return res
