import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor

from GaussianProcessTools.parallel import point_generator
from GaussianProcessTools import optimization as opt


def func(x):
    return 5 - x**2

def acq(x,*args):
    return -opt.upper_confidence_bound(x.reshape(-1,1),*args)

x = np.array((2,0,-2)).reshape(-1,1)
y = func(x)

gpr = GaussianProcessRegressor()
gpr.fit(x,y)

t = np.linspace(-3,3).reshape(-1,1)
mu,std = gpr.predict(t,return_std=True)
mu = mu.flatten()
t = t.flatten()
fig,ax = plt.subplots()
ax.plot(x,y,'+')
ax.plot(t,mu,c='C1')
ax.fill_between(t,(mu-std),(mu + std),fc='C1',lw=0,alpha=0.5)

bounds = np.array((-3,3)).reshape(-1,2)
GPRs = [gpr]
args = gpr
n = 3
pts = point_generator.generate_points(GPRs,bounds,acq,n,args = gpr)
print(pts)

plt.show()
