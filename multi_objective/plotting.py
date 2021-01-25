import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon

from . import pareto

#plt.style.use('PRAB_style.mplstyle')

def plot_full(model, f_ground = None, n = 20):

    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.labelpad'] = 0.0
    mpl.rcParams['figure.subplot.top'] = 0.975
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.975
    mpl.rcParams['figure.subplot.hspace'] = 0.375
    mpl.rcParams['figure.subplot.wspace'] = 0.375
    mpl.rcParams['figure.figsize'] = 3.75, 4
    if callable(f_ground):
        add = 1
    else:
        add = 0
        
    fig, axes = plt.subplots(2 + add,2)
    fig.set_size_inches(3.75,4)
    
    assert model.domain_dim == 2
        
    PF = model.get_PF()
    fargs = [model]    

    
    x = np.linspace(*model.bounds[0,:],n)
    y = np.linspace(*model.bounds[1,:],n)
    xx, yy = np.meshgrid(x,y)
    pts = np.vstack((xx.ravel(),yy.ravel())).T

    g = []
    f_gnd = []
    for pt in pts:
        g += [model.acq(pt.reshape(1,-1), model)]
        if add:
            f_gnd += [f_ground(pt)]

    f_gnd = np.array(f_gnd)
            
    f1 = model.GPRs[0].predict_y(pts)[0]
    f2 = model.GPRs[1].predict_y(pts)[0]


    
    if add:
        faxes = [*axes[0], axes[add,0],axes[add,1],axes[add + 1,0]]
        funcs = [*f_gnd.T, f1,f2,g]

    else:
        faxes = [axes[add,0],axes[add,1],axes[add + 1,0]]
        funcs = [f1,f2,g]

    for ax, fun in zip(faxes,funcs):
        fun = np.array(fun).reshape(n,n)
    
        c = ax.pcolor(xx,yy,fun/(np.max(fun) - np.min(fun)),cmap = 'cividis')
        #ax.figure.colorbar(c,ax=ax)          

    for ax in faxes[:2]:
        ax.plot((-1,1),(-1,1),c='C2')
        
    for ax in faxes[2:]:
        ax.plot(*model.get_data('X').T,'r.')
        #ax.set_yticks([])
        #ax.set_xticks([])
        
    axes[-1,0].set_xlabel('$x_1$')
    axes[-1,0].set_ylabel('$x_2$')
    axes[-1,0].set_yticks([-2,0,2])
    axes[-1,0].set_xticks([-2,0,2])
        
    obj_ax = axes[add + 1,1]
    obj_ax.plot(*model.get_data('Y').T,'r.',label='Samples')
    obj_ax.plot((2*np.sqrt(2),0),(0,2*np.sqrt(2)))
    obj_ax.set_xlabel('$f_1$')
    obj_ax.set_ylabel('$f_2$')
    #fig.colorbar(cax)

    for a in axes.flatten()[:4]:
        a.set_xticks([-2,0,2])
        a.set_yticks([-2,0,2])
        a.set_xlabel('$x_1$')
        a.set_ylabel('$x_2$')
        
    axes = axes.flatten()
    l = ['a','b','c','d','e','f']
    for a, label in zip(axes,l):
        a.text(-0.15,1.05,f'({label})',ha = 'right',
               va = 'top', transform = a.transAxes,
               fontdict = {'size':10})#,
               #bbox = {'facecolor': 'white',
               #        'alpha': 0.85,
               #        'pad': 0.25,
               #        'boxstyle':'round'})
    return fig

               
def plot_acq(self, ax = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    assert self.domain_dim == 2
        
    self.PF = self.get_PF()
    fargs = [self]    

    n = kwargs.get('n',30)
    x = np.linspace(*self.bounds[0,:],n)
    y = np.linspace(*self.bounds[1,:],n)
    xx, yy = np.meshgrid(x,y)
    pts = np.vstack((xx.ravel(),yy.ravel())).T

    f = []
    for pt in pts:
        f += [self.obj(pt,*fargs)]

    f = np.array(f).reshape(n,n)
    
    c = ax.pcolor(xx,yy,f,cmap='cividis')
    ax.figure.colorbar(c,ax=ax)          
    
    ax.plot(*self.get_data('X').T,'+')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
        
    return ax

def plot_constr(self, ax = None,**kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        
    fargs = [self.GPRs,self.PF,self.A,self.B]    

    n = kwargs.get('n',30)
    x = np.linspace(*self.bounds[0,:],n)
    y = np.linspace(*self.bounds[1,:],n)
    xx, yy = np.meshgrid(x,y)
    pts = np.vstack((xx.ravel(),yy.ravel())).T
    
    f = []
    for pt in pts:
        f += [self.constraints[0].predict(np.atleast_2d(pt))]

    f = np.array(f).reshape(n,n)
    
    c = ax.pcolor(xx,yy,f, cmap = 'cividis')#,cmap='plasma')
    cbar = ax.figure.colorbar(c,ax=ax)          
    #cbar.set_lim(0,1)
    
    if kwargs.get('change_initial',False):
        X_i = self.get_data('X')[:10]
        X_m = self.get_data('X')[10:]

        ax.plot(*X_i.T,'+C1',label = 'Initial')
        ax.plot(*X_m.T,'.C2',label = 'Selected')
        
    else:
        X_feas = self.get_data('X',valid = True)
        X_nonfeas = self.get_data('X',valid = False)
        
        ax.plot(*X_feas.T,'+r')
        ax.plot(*X_nonfeas.T, 'or')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
        
    return ax, cbar


def create_pf_patch(pf,ref):

    assert pf.shape[1] == 2
    pf = pareto.sort_along_first_axis(pf)
    pf = np.vstack((ref,pf))
    v = []
    print(pf)
    for i in range(len(pf)-1):
        v += [pf[i]]
        v += [[pf[i][0],pf[i+1][1]]]


    v += [pf[i+1]]
    v += [[pf[i+1][0],ref[1]]]
    
    return Polygon(np.array(v),True)
