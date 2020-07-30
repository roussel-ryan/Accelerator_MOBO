import numpy as np
import pygmo as pg

def get_UHVI(X, GPRs, PF, A, B, beta = 0.01):
    '''computes the UCB Hypervolume improvement

    Parameters
    ----------
    
    X : ndarray, shape (n,m)
        Array of n points to calculate the UHVI at

    GPRs : list
        List GPFlow regressor objects

    PF : ndarray, shape (o,m)
        Array of o non-dominated, sorted points, not checked

    A : ndarray, shape (m,)
        Lower bound of hypervolume

    B : ndarray, shape (m.)
        Upper bound of hypervolume (also known as the reference point)

    beta : float, optional
        Tradeoff parameter that determines exploitation (beta << 1) vs 
        exploration (beta >> 1), default = 0.01

    Returns
    -------
    uhvi : ndarray (n,)
        UHVI of each input point

    '''

    if not len(X.shape) == 2:
        X = np.atleast_2d(X)
    
    n_pts = X.shape[0]
    n_obj = len(GPRs)
    uhvi = np.empty(n_pts)

    for i in range(n_pts):
        #using the GPRs, compute the UHVI point f = mu - beta * std
        uhvi_pt = np.empty(n_obj)

        for j in range(n_obj):
            mu, std = GPRs[j].predict_f(np.atleast_2d(X[i]))
            uhvi_pt[j] = (mu - beta * std).numpy()

        
        #add the uhvi_pt to the list of points
        points = np.vstack((PF,np.atleast_2d(uhvi_pt)))
        hv = pg.hypervolume(points)

        #calculate the exclusive contribution to the hypervolume from our point
        uhvi[i] = hv.exclusive(len(points)-1, B)

    return uhvi
            
    
    
