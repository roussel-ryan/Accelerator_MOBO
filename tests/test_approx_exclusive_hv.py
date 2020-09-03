import numpy as np
import matplotlib.pyplot as plt

import pygmo as pg

import time

from GaussianProcessTools.multi_objective import pareto

def project(x, PF):
    #project points from PF onto x
    proj_pf = []
    print(x)
    for pt in PF:
        proj_pf += [np.where(x > pt, x, pt)]

    return np.array(proj_pf)

def approx_exclusive_hv(x, projected_pf, ref):
    
    fpras = pg.bf_fpras(eps=0.1, delta=0.1)
    
    hv = pg.hypervolume(projected_pf)
    return np.prod(ref - x) - hv.compute(ref, hv_algo=fpras)

def exclusive_hv(x, pf, ref):
    pts = np.vstack((pf,np.atleast_2d(x)))

    hv = pg.hypervolume(pts)
    return hv.exclusive(len(pts)-1,ref)


def main():
    dim = 10
    npts = 120

    ref = np.ones(dim)
    np.random.seed(1)
    pts = np.random.uniform(0.0,1.0,(npts,dim))
    #pts = np.vstack((pts,np.atleast_2d(ref*0.1)))
    
    #x = np.linspace(0.1,0.9,7)
    #y = 1 - x
    
    #PF = np.stack((x,y)).T

    #fig,ax = plt.subplots()
    #ax.plot(*PF.T,'+')
    #ax.plot(*ref.T,'+')

    xs = 0.25*ref

    ndf, _, _, _ = pg.fast_non_dominated_sorting(pts)
    print(f'npts on PF {len(ndf[0])}')
    
    print('approx')
    start = time.time()
    print(f'vol: {run_approx(xs, pts, ref)}')
    print(f'time {time.time() - start}s')

    print('exact')
    start = time.time()
    print(f'vol: {exclusive_hv(xs, pts, ref)}')
    print(f'time {time.time() - start}s')


    
def run_approx(xs,pts,ref):
    #check if point xs is dominated by any other point
    is_dominated = False
    for pt in pts:
        if np.all(pt <= xs):
            is_dominated = True
            break

    if is_dominated:
        return 0.0
    else:
        #ndf, _, _, _ = pg.fast_non_dominated_sorting(pts)
        #pts = pts[ndf[0]]
        
        proj_pf = project(xs,pts)
        #proj_pf = pareto.get_non_dominated_set(proj_pf)
        
        ndf, _, _, _ = pg.fast_non_dominated_sorting(proj_pf)
        proj_pf = proj_pf[ndf[0]]
    
        return approx_exclusive_hv(xs, proj_pf, ref)
    
main()
plt.show()

    
