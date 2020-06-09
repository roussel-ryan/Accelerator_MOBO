import numpy as np
import matplotlib.pyplot as plt

from GaussianProcessTools.lineBO import optimizer
from GaussianProcessTools.lineBO import oracles
import scipy.optimize as opt

import logging
logging.basicConfig(level=logging.INFO)

def hartmann6D(x):
    alpha = np.array((1.0,1.2,3.0,3.2))
    A = np.array(((10.,3.,17.,3.5,1.7,8.),
                  (0.05,10.,17.,0.1,8.,14.),
                  (3.,3.5,1.7,10.,17.,8.),
                  (17.,8.,0.05,10.,0.1,14.)))
    P = 1e-4 * np.array(((1312,1696,5569,124,8283,5886),
                         (2329,4135,8307,3736,1004,9991),
                         (2348,1451,3522,2883,3047,6650),
                         (4047,8828,8732,5743,1091,381)))
    inn = (x - P)**2
    #print(inn)
    inner = - np.sum(A * inn,axis=1)
    return -np.sum(alpha.T * np.exp(inner))


def obj(x):
    #return hartmann6D(x)
    #return (4 - 2.1*x[0]**2 + x[0]**4 / 3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2


    #return opt.rosen(x)
    return np.linalg.norm(x - np.array((0.5,0.5)))

bounds = np.array(((-3,3),(-2,2)))
#print(obj(np.array((0.2,0.15,0.47,0.27,0.31,0.65))))

#bounds = np.vstack((np.zeros(6),np.ones(6))).T
#x0 = np.random.uniform(len(bounds))
x0 = np.array((-1,-1)).reshape(-1,2)
logging.info(f'domain bounds: {bounds}')
lineBO = optimizer.LineOpt(bounds,obj,x0 = x0,verbose=0)

fig,ax = plt.subplots()
lineBO.optimize()

ax.plot(np.array(lineBO.f).flatten())

plt.show()

