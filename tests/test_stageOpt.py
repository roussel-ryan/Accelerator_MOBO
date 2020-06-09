import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
import logging

from GaussianProcessTools.stageopt import optimizer as stageOpt
from GaussianProcessTools.lineBO import optimizer as lineOpt

def ofunc(x):
    return 5 - x**2

def cfunc1(x):
    return 2 - (x + 0.5)**2

def cfunc2(x):
    return ((1 / (1 + np.exp(x))) - 0.5)*2

def test1D():
    inputs = {}
    inputs['bounds'] = np.array((-5,5)).reshape(-1,2)
    kernel = RBF(1.0,'fixed')
    inputs['gprf'] = GaussianProcessRegressor(kernel,alpha = 0.01)
    inputs['gprc'] = [GaussianProcessRegressor(kernel,alpha = 0.01),GaussianProcessRegressor(kernel,alpha = 0.01)]

    inputs['X0'] = np.array((-1.1)).reshape(-1,1)
    inputs['ofun'] = ofunc
    inputs['cfun'] = [cfunc1,cfunc2]
    
    inputs['Y0'] = ofunc(inputs['X0'])
    inputs['C0'] = np.array([f(inputs['X0']) for f in inputs['cfun']]).T[0]
    inputs['h'] = np.zeros(len(inputs['cfun']))
    inputs['n_grid'] = 50
    inputs['T0'] = 5
    inputs['verbose'] = False
    
    opt = optimizer.StageOpt(inputs)


    

def main():
    test1D()

    
    

main()
plt.show()
