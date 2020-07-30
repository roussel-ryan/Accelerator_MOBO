import numpy as np
import matplotlib.pyplot as plt

from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var = 6,
                         n_obj = 1,
                         xl = -np.ones(6),
                         xu = np.ones(6))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:,0]**2 + x[:,1]**2

        out["F"] = np.column_stack([f1])

problem = MyProblem()

algorithm = NSGA2(
    pop_size=10,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

termination = get_termination('n_gen',40)
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)

print(res.X)
