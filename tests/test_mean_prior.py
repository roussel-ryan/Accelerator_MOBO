import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gpflow

from GaussianProcessTools import priors

def f(x):
    return np.sin(x)

def prior(x):
    return 5*np.sin(x)

def main():
    n = 3
    x = np.random.uniform(0,2*np.pi,n).reshape(-1,1)
    y = f(x)


    k = gpflow.kernels.RBF()
    mean_prior = priors.CustomPrior(prior)

    m = gpflow.models.GPR(data = (x,y),
                          kernel=k,
                          mean_function = mean_prior)
    m.likelihood.variance.assign(1.0e-5)
    
    g = np.linspace(0,2*np.pi).reshape(-1,1)
    p = m.predict_y(g)[0].numpy()
    s = np.sqrt(m.predict_y(g)[1].numpy())
    
    fig,ax = plt.subplots()
    ax.plot(g,prior(g),label='prior',c='C0')
    ax.plot(g,p,label='posterior',c='C1')
    ax.fill_between(g.flatten(),(p-s).flatten(),(p+s).flatten(),
                    lw=0,color='C1',alpha=0.25)
    
    ax.plot(g,f(g),label='ground',c='C2')
    ax.plot(x,y,'+',label='observations',c='C3')
    ax.legend()
    
    
    print(y)

main()
plt.show()    

    
    
