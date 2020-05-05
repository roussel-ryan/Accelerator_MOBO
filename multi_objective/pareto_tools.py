import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def dominates(a,b,strict = False):
    '''returns True if a dominates over b (considering minimization problems)'''
    #a is no worse than b for all objectives
    cond1 = np.all(a <= b)
    
    #a is better than b for at least one objective
    cond2 = np.any(a < b)

    if strict:
        #for strict dominance all elements have to be better
        return np.all(a < b)
    else:
        return np.all((cond1,cond2))

def dominates_subset(a,pset,strict = False):
    '''returns the subset of <set> that <a> dominates'''
    subset = []
    for ele in pset:
        if dominates(a,ele,strict):
            subset.append(ele)
    return np.array(subset)
    
def is_dominated(pt,pset,strict = False):
    '''Returns true if pt is dominated by any points in pset'''
    is_domed = False
    for i in range(len(pset)):
        #if any point in pset dominates over pt then pt is dominated
        if dominates(pset[i],pt,strict):
            is_domed = True
    return is_domed

def in_set(pt,s):
    ''' test if point <pt> is in set <s>'''
    return np.any(np.equal(s,pt).all(1))

    
def get_non_dominated_set(s):
    '''implimentation of Kung's Algorithm'''
        
    def _front(P):
        '''see https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf'''
        
        if len(P) == 1:
            return P
        else:
            P_list = np.array_split(P,2)
            T = _front(P_list[0])
            B = _front(P_list[1])

            M = T
            for i in range(len(B)):
                is_dom = is_dominated(B[i],T)

                #is_dom = False
                #for j in range(len(T)):
                #    #if T[j] dominates over B[i] then B[i] is dominated 
                #    if dominates(T[j],B[i]):
                #        is_dom = True

                if not is_dom:
                    M = np.vstack((M,B[i]))
            return M
    #sort s along first objective
    ind = np.argsort(s.T[0])
    
    P = s[ind]
    #logging.info(P)
    return _front(P)

def sort_along_first_axis(s):
    ind = np.argsort(s.T[0])
    return s[ind]

def main():
    n = 50
    s = np.random.uniform(size=(n,2))
    fig,ax = plt.subplots()
    ax.plot(*s.T,'+')

    P = get_non_dominated_set(s)

    ax.plot(*P.T,'.')

    logging.info(in_set(np.array([0,0]),P.T))

if __name__=='__main__':
    main()
    plt.show()
