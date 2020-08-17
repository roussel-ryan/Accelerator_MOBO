import numpy as np
import matplotlib.pyplot as plt
import logging


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
