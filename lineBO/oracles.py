import numpy as np

def random(dim):
    ''' returns random unit vector to determine direction'''
    d = np.random.uniform(-1,1,size = dim)
    return d / np.linalg.norm(d)
