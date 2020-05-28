import numpy as np
import matplotlib.pyplot as plt

from GaussianProcessTools.multi_objective import EI_tools
from GaussianProcessTools.multi_objective import pareto_tools as PT_tools

def main():
    x = np.linspace(0,1,10)
    y = 1.0 - x

    F = np.vstack((x,y)).T
    ob = np.array((0.25,0.25))
    F = np.vstack((F,ob))
    
    r = np.array((1.,1.))
    A = np.array((0.0,0.0))
    B = r
    print(A)
    print(B)

    n = 20
    xx, yy = np.meshgrid(np.linspace(0,2,n),np.linspace(0,2,n))
    pts = np.vstack((xx.ravel(),yy.ravel())).T

    F = PT_tools.get_PF(F)
    F = PT_tools.sort_along_first_axis(F)
    
    evhi = []
    for pt in pts:
        evhi.append(EI_tools.EHVI_2D(pt,np.ones(2)*0.15,F,r,A,B))
    pt = np.array((0.9,0.9))
    #EI_tools.EHVI_2D(pt,np.ones(2)*0.00000001,F,r,A,B)
    
    ehvi = np.array(evhi).reshape((n,n))
    #print(ehvi)
    
    fig,ax = plt.subplots()
    ax.plot(*F.T,'+')

    c = ax.pcolor(xx,yy,ehvi)
    fig.colorbar(c)
    
main()
plt.show()
