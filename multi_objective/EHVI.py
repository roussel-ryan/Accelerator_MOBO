import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from scipy.stats import norm
import logging

#p = [mu,sigma,A,B]
def Z(p):
    mu,sigma,A,B = p
#    logging.info((A,B))
    if sigma == 0.0 and (mu <= A or B <= mu):
        return 0
    else:
        return norm.cdf(np.divide(B - mu,sigma)) - norm.cdf(np.divide(A - mu,sigma))
   
def phi(x,p):
    mu,sigma,A,B = p
    if x <= A or B <= x:
        return 0.0
    else:
        return Z(p) * np.divide(norm.pdf(np.divide(x - mu,sigma)),sigma)

def PHI(x,p):
    mu,sigma,A,B = p
    if x <= A:
        return 0
    elif (A < x)*(x < B):
        return Z(p) * (norm.cdf(np.divide(x - mu,sigma)) - 
                       norm.cdf(np.divide(A - mu,sigma)))
    else:
        return 1

def PSI(a,b,p):
    mu,sigma,A,B = p
    val = Z(p) * (sigma*norm.pdf(np.divide(b - mu,sigma)) +\
                      (a - mu)*norm.cdf(np.divide(b - mu,sigma)) -\
                      (sigma*norm.pdf(np.divide(A - mu,sigma)) + \
                       (a - mu)*norm.cdf(np.divide(A - mu,sigma))))
    return val

def get_EHVI(X,GPRs,S,B,A = None):
    '''
    x: point input from optimizer
    S: set of sorted, nondominated observed points
    GPRs: list of GP regressors
    r: reference point
    '''

    if A is None:
        A = np.zeros(2)
    
    dim = len(X)

    f = np.array([ele.predict(X.reshape(-1,dim)) for ele in GPRs]).reshape(2,2).T
    ehvi = -EHVI_2D(f[0],f[1],S,B,A)
    #logging.info((f[0],f[1],ehvi))
    return ehvi

def EHVI_2D(mu,sigma,Y,B,A,verbose = False):
    #A = kwargs.get('A',np.zeros(2))
    #B = kwargs.get('B',r)
    
    #number of non-dominated points
    n = len(Y)

    #make sure that the points are sorted along first axis in decending order
    #logging.info(f'f1: {Y.T[0]}')
    assert np.all(np.diff(Y.T[0]) <= 0)
    
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
