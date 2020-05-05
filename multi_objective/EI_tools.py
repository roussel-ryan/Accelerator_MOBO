import numpy as np
from scipy.stats import norm

def PSI(a,b,mu,sigma):
    try:
        s = (b - mu) / sigma
        return sigma * norm.pdf(s) + (a - mu)*CDF(b,mu,sigma)
    except ZeroDivisionError:
        return (a - mu)*CDF(b,mu,sigma)
        
def CDF(x,mu,sigma):
    try:
        s = (x - mu) / sigma
        return norm.cdf(s)
    
    except ZeroDivisionError:
        if x > mu:
            return 1.0
        else:
            return 0.0
        
def 2D_EHVI(mu,sigma,Y,r):
    #number of non-dominated points
    n = len(Y)
    
    #add bounding points to set
    Y = np.vstack(((r[0], -np.inf),Y,(-np.inf,r[1])))
    #logging.info(Y)

    sum1 = 0
    sum2 = 0
    for i in range(1,n+1):
        sum1 = sum1 + (Y[i-1][0] - Y[i][0])*CDF(Y[i][0],mu[0],sigma[0])*PSI(Y[i][1],Y[i][1],mu[1],sigma[1])
        sum2 = sum2 + (PSI(Y[i-1][0],Y[i-1][0],mu[0],sigma[0]) - PSI(Y[i-1][0],Y[i][0],mu[0],sigma[0])) * PSI(Y[i][1], Y[i][1], mu[1],sigma[1]) 
        
    return sum1 + sum2
