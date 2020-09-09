import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import time
import gpflow


def main():
    dim = 7
    npts = 400
    x0 = np.random.uniform(size=(npts,dim))
    y0 = x0**2

    #print(x0)
    #print(y0)
    
    k = gpflow.kernels.RBF(lengthscales = 0.5, variance = 0.5)
    gpr = gpflow.models.GPR((x0,y0), kernel = k, noise_variance = 1e-5)
    x = np.zeros(dim).reshape(-1,dim)
   
    start = time.time()
    gpr.predict_y(x)
    print(f'{time.time() - start}')


    


main()
