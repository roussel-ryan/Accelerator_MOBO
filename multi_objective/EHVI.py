import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import scipy
import logging

from . import pareto
#p = [mu,sigma,A,B]

def cdf(s):
    return 0.5*(1 + scipy.special.erf(s/np.sqrt(2)))

def pdf(s):
    return np.exp(-s**2 / 2) / (np.sqrt(2)*np.pi)

def Z(p):
    mu,sigma,A,B = p
#    logging.info((A,B))
    if sigma == 0.0 and (mu <= A or B <= mu):
        return 0
    else:
        return cdf(np.divide(B - mu,sigma)) - cdf(np.divide(A - mu,sigma))
   
def phi(x,p):
    mu,sigma,A,B = p
    if x <= A or B <= x:
        return 0.0
    else:
        return Z(p) * np.divide(pdf(np.divide(x - mu,sigma)),sigma)

def PHI(x,p):
    mu,sigma,A,B = p
    if x <= A:
        return 0
    elif (A < x)*(x < B):
        return Z(p) * (cdf(np.divide(x - mu,sigma)) - 
                       cdf(np.divide(A - mu,sigma)))
    else:
        return 1

def PSI(a,b,p):
    mu,sigma,A,B = p
    val = Z(p) * (sigma*pdf(np.divide(b - mu,sigma)) +\
                      (a - mu)*cdf(np.divide(b - mu,sigma)) -\
                      (sigma*pdf(np.divide(A - mu,sigma)) + \
                       (a - mu)*cdf(np.divide(A - mu,sigma))))
    return val

def get_EHVI(X, GPRs, PF, A, B):
    '''
    x: input points from optimizer, must be 2D
    S: set of sorted, nondominated observed points
    GPRs: list of GP regressors
    r: reference point
    '''

    
    if A is None:
        A = np.zeros(2)

    if not len(X.shape) == 2:
        X = np.atleast_2d(X)
        
    ehvi = np.empty(len(X))
    for i in range(len(X)):
        dim = len(X[i])
        
        f = np.array([ele.predict_f(X[i].reshape(-1,dim)) for ele in GPRs]).reshape(2,2).T
        ehvi[i] = EHVI_2D(f[0], f[1], PF, A, B)
    return ehvi.reshape(-1,1)

def EHVI_2D(mu,sigma,Y,A,B,verbose = False):
    #A = kwargs.get('A',np.zeros(2))
    #B = kwargs.get('B',r)
    
    #number of non-dominated points
    n = len(Y)

    #make sure that the points are sorted along first axis in decending order
    #logging.info(f'f1: {Y.T[0]}')

    if not np.all(np.diff(Y.T[0]) <= 0):
        Y = pareto.sort_along_first_axis(Y)
    
    #add bounding points to set
    Y = np.vstack(((B[0], A[1]),Y,(A[0],B[1])))
    #logging.info(len(Y))

    sum1 = 0
    sum2 = 0
    #limits and GP results for each dimention
    p = np.vstack((mu,sigma,A,B)).T
    #print(p)
    
    
    for i in range(1,n+2):
        if verbose:
            logging.info(f'Summation stats for rectangle {i}')
            logging.info(f'mean coords: {mu}')
            logging.info(f'Rectangle coordinates: Y[i-1] = {Y[i-1]},Y[i] = {Y[i]}')
            logging.info(f'PHI(Y[i][0],p[0]): {PHI(Y[i][0],p[0])}')
            logging.info(f'PSI(Y[i][1],Y[i][1], p[1]): {PSI(Y[i][1],Y[i][1], p[1])}')
        #change index i such that the paper index matches python indexing
        #j = i - 1

        term1 = (Y[i-1][0] - Y[i][0])*PHI(Y[i][0],p[0])*PSI(Y[i][1],Y[i][1],p[1])        
        term2 = (PSI(Y[i-1][0],Y[i-1][0],p[0]) - PSI(Y[i-1][0],Y[i][0],p[0])) * PSI(Y[i][1],Y[i][1], p[1])

        if verbose:
            logging.info(f'term1 {term1}')
            logging.info(f'term2 {term2}')
        
        sum1 = sum1 + term1
        sum2 = sum2 + term2
    return sum1 + sum2
