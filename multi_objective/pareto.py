import numpy as np
import matplotlib.pyplot as plt
import logging

import pygmo as pg


def get_PF(data, ref, return_contr = False, tol=1e-8,low_ref = None):
    '''get the pareto front set for multi-objective minimization

    there is an issue with pygmo.fast non dominated sorting,
    where points are listed on the pareto front but do not contribute any hypervolume to the PF HV
    this function acts as a wrapper that fixes the issue
    
    Optional Parameters
    -------------------

    return_contr : bool, optional
        If true get_PF returns the exclusive hypervolume contribution for 
        each point in the pareto set

    '''

    if not np.any(low_ref):
        low_ref = -np.inf * np.ones_like(data[0])

    #remove points that are not in the target domain
    F = []
    for pt in data:
        if not (np.any(pt < low_ref) or np.any(pt > ref)):
            F += [np.atleast_2d(pt)]

    data = np.vstack(F)
    
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(data)
    pf_idx = ndf[0].astype(int)
    data = data[pf_idx]

    hv    = pg.hypervolume(data)
    contr = hv.contributions(ref)

    front = data[np.argwhere(contr > tol).flatten()]
    contr = contr[np.argwhere(contr > tol).flatten()]

    ind = np.argsort(front.T[0])
    front = front[ind][::-1]
    contr = contr[ind][::-1]
    
    if return_contr:
        return front, contr
    else:
        return front

def sort_along_first_axis(s):
    ind = np.argsort(s.T[0])
    return s[ind][::-1]

def main():
    #testing
    n = 50
    s = np.random.uniform(size=(n,2))
    fig,ax = plt.subplots()
    ax.plot(*s.T,'+')

    P = get_non_dominated_set(s)
    logging.info(get_PF_indicies(s))
    ax.plot(*P.T,'.')


if __name__=='__main__':
    main()
    plt.show()
