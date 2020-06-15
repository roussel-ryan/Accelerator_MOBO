import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from GaussianProcessTools import optimizers

#def f(x,*args):
#    return np.linalg.norm(x - np.ones(2))**2 + np.linalg.norm(x + np.ones(2))**2 

#def f(x,*args):
#    return 2.0*(x[0] - 1)**2 - 1.05*(x[0] - 1)**4 + (x[0]-1)**6 / 6 + (x[0]-1)*x[1] +x[1]**2

def f(x,*args):
    return np.sin(x[0]) * np.sin(x[1])

def do_optimization(x):
    bounds = np.array(((-5,5),(-5,5)))
    opt = optimizers.LineOpt()
    res = opt.minimize(bounds,f,x)
    print(res.x)

def basic_test():
    x0 = np.random.uniform(-5,5,(10,2))

    procs = []
    for ele in x0:
        proc = multiprocessing.Process(target=do_optimization, args=(ele,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

def adv_test():
    bounds = np.array(((-5,5),(-5,5)))
    popt = optimizers.ParallelLineOpt()
    res = popt.minimize(bounds,f)
    
if __name__ == '__main__':
    adv_test()
