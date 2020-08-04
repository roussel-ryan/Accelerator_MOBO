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

        algo = pg.algorithm(pg.pso(gen = self.generations))
       
        pop = pg.population(problem,size = self.pop_size)
        
        self.logger.info('Starting swarm optimization')
        pop = algo.evolve(pop)
        self.logger.info('Done with swarm optimization')

        #create result object
        res = base.Result(pop.get_x()[pop.best_idx()], pop.get_f()[pop.best_idx()])
        
        return res

class ParallelSwarmOpt(base.BlackBoxOptimizer):
    def __init__(self,**kwargs):
        #swarm optimization hyperparameters
        self.generations = kwargs.get('generations',20)
        self.pop_size    = kwargs.get('pop_size', 50)
        self.islands   = kwargs.get('islands',1)

        self.logger            = logging.getLogger(__name__)

        
    def minimize(self, bounds, func, args = [], x0 = None):
        p = OptProblem(bounds, func, args)
        problem = pg.problem(p)

        algo = pg.algorithm(pg.pso(gen = self.generations))
        #archi = pg.archipelago(n = self.islands,
        #                       algo = algo,
        #                       prob = problem,
        #                       pop_size = self.pop_size)

        pop = pg.population(problem,size = self.pop_size)
        
        self.logger.info('Starting swarm optimization')
        #self.logger.info(archi)
        pop.evolve()
        #archi.wait()
        #self.logger.info(archi)
        #self.logger.info('Done with swarm optimization')

        #champs_f = np.array(archi.get_champions_f()).flatten()
        #best_champ_indx = np.argmin(champs_f)
        #best_champ_x = archi.get_champions_x()[best_champ_indx]
        #best_champ_f = archi.get_champions_f()[best_champ_indx]
        
        #create result object
        res = base.Result(best_champ_x,best_champ_f)
        
        return re
