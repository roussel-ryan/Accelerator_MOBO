import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import copy
import pickle

from scipy.stats import multivariate_normal

#suppress output messages below "ERROR" from tensorflow
#and prevent the use of any system GPU's
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import gpflow

from GaussianProcessTools import mobo
from GaussianProcessTools.optimizers import evolutionary
from GaussianProcessTools import infill
from GaussianProcessTools.multi_objective import plotting
from GaussianProcessTools.multi_objective import pareto

import tensorflow as tf
import logging




class MyAnimation:
    def __init__(self, bounds, model, axes, tsteps = 50):
        logging.info('init')
        self.model = model
        self.bounds = bounds
        self.axes = axes

        #initialize plotting grid
        self.n = 75
        self.x = np.linspace(*bounds[0],self.n)
        self.xx = np.meshgrid(self.x,self.x)
        self.pts = np.vstack([ele.ravel() for ele in self.xx]).T

        self.t = np.linspace(0, 400, tsteps)
        
        #initialize meshes
        self.quads = []
        for ax in axes:
            self.quads += [ax.pcolor(*self.xx,
                                     np.zeros((self.n,self.n)),
                                     vmin = -5,vmax = 0)]

        #initialize points
        self.obs_lines = []
        for ax in axes:
            ln, = ax.plot([],[],'+r')
            self.obs_lines += [ln]

        #ideal lines
        self.ideal_lines = []
        for ax in axes:
            ln, = ax.plot([],[],'-C1')
            self.ideal_lines += [ln]

        self.axes[0].set_ylabel('$x_2$')
        self.axes[2].set_ylabel('$x_2$')
        self.axes[2].set_xlabel('$x_1$')
        self.axes[3].set_xlabel('$x_1$')
        
            
        
    def transform_for_pcolor(self,arr):
        return arr.reshape(self.n,self.n)[:-1,:-1].ravel()
            
    def update(self,i):
        logging.info(i)
        f1 = []
        f2 = []
        gp_f1 = []
        gp_f2 = []

        X = np.hstack((self.pts,np.ones((len(self.pts),1))*self.t[i]))
        
        for pt in self.pts:
            F = f(pt, self.t[i])
            f1 += [F[0]]
            f2 += [F[1]]

        pred1 = self.model.GPRs[0].predict_y(X)
        pred2 = self.model.GPRs[1].predict_y(X)

        gp_f1 = pred1[0].numpy().flatten()
        gp_f2 = pred2[0].numpy().flatten()
            

        f1 = self.transform_for_pcolor(np.array(f1))
        f2 = self.transform_for_pcolor(np.array(f2))
        gp_f1 = self.transform_for_pcolor(np.array(gp_f1))
        gp_f2 = self.transform_for_pcolor(np.array(gp_f2))
        
        
        self.quads[0].set_array(f1)
        self.quads[1].set_array(f2)
        self.quads[2].set_array(gp_f1)
        self.quads[3].set_array(gp_f2)

        #plot input points
        X = self.model.get_data('X', time = self.t[i])[::2]
        #print(X)
        for ln in self.obs_lines:
            ln.set_data(*X[-25:].T)

        #plot input points
        ideal_pts = np.array(f(pt,self.t[i],True)).T
        for ln in self.ideal_lines:
            ln.set_data(*ideal_pts)


        self.axes[0].set_title(f'$f_1(x,{int(self.t[i])})$')
        self.axes[1].set_title(f'$f_2(x,{int(self.t[i])})$')
        self.axes[2].set_title('$\mathcal{GP}_1(x)$')
        self.axes[3].set_title('$\mathcal{GP}_2(x)$')

            
def f(x, t, get_centers = False):
    T = 200
    theta = 2 * np.pi * t / T + np.pi/4
    x01 = np.array((np.cos(theta),np.sin(theta))) * 0.5
    x02 = np.array((np.cos(theta),np.sin(theta))) * -0.5
    #x01 = np.array((1,1)) * 0.5 * np.cos(np.pi/4)
    #x02 = np.array((-1,-1)) * 0.5 * np.cos(np.pi/4)

    f1 = np.linalg.norm(x - x01) - 5.0
    f2 = np.linalg.norm(x - x02) - 5.0

    if get_centers:
        return [x01,x02]
    else:
        return np.array([f1, f2])




def plot_objective_space_density(model, t):
    n = 500
    x = np.linspace(-5, -2.5, n)
    xx = np.meshgrid(x,x)
    pts = np.vstack([ele.ravel() for ele in xx]).T
    #print(pts)
    
    X = model.get_data('X', time = t)
    Y = model.get_data('Y', time = t)
    X = np.hstack([X, np.ones(len(X)).reshape(-1,1)*t])
    
    F = np.zeros(len(pts))

    
    PCB = []
    m_vecs = []
    for x in X:
        m_vec = []
        s_vec = []

        for i in range(model.obj_dim):
            m, s = model.GPRs[i].predict_f(np.atleast_2d(x))
            m_vec += [m.numpy()]
            s_vec += [s.numpy()]

        
        s_vec = np.array(s_vec).flatten()
        m_vec = np.array(m_vec).flatten()

        PCB += [m_vec + np.sqrt(s_vec)]
        m_vecs += [m_vec]
        rv = multivariate_normal(m_vec,np.diag(s_vec))

        
        F = F + rv.pdf(pts)

    PCB = np.array(PCB)

    PCB_PF = pareto.get_PF(PCB,model.B, low_ref = model.A)

    m_vecs = np.array(m_vecs)
    fig,ax = plt.subplots()
    c = ax.pcolor(*xx, np.log(F).reshape(n,n),cmap='Greys',vmin = 0, vmax = 11,zorder=0)
    ax.plot(*Y.T,'+')
    ax.plot(*PCB.T,'+')
    ax.plot(*PCB_PF.T,'C1')

    
    
    #for i in np.arange(len(Y))[::10]:
    #    l = np.vstack((Y[i],m_vecs[i]))
        #l = np.vstack((Y[i],PCB[i]))
    #    l2 = np.vstack((PCB[i],m_vecs[i]))
    #    ax.plot(*l.T,'C2',zorder = 1)
        #ax.plot(*l2.T,'C3',zorder = 1)

    #for i in np.arange(len(Y))[:10]:
    #    l = np.vstack((Y[i],m_vecs[i]))
    #    #l = np.vstack((Y[i],PCB[i]))
    #    l2 = np.vstack((PCB[i],m_vecs[i]))
    #    ax.plot(*l.T,'C2',zorder = 1)
    #    ax.plot(*l2.T,'C3',zorder = 1)

    ax.set_ylim(-5,-2.5)
    ax.set_xlim(-5,-2.5)
    ax.set_title(f'Probability density at time: {t}')
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')

    
    cax = fig.colorbar(c)
    cax.set_label('$\log_{10}(P(f_1,f_2,t))$')

    fig.savefig(f'drift_{model.time}.png')

def plot_time_dep(m,i,x,T):

    X = []
    gnd = []
    for t in T:
        X += [np.append(x,t).reshape(-1,3)]
        gnd += [f(x,t)[i]]
    X = np.vstack(X)

    F = m.GPRs[i].predict_y(X)
    mean = F[0].numpy().flatten()
    s = np.sqrt(F[1].numpy().flatten())

    
    fig,ax = plt.subplots()
    ax.plot(T.flatten(),mean)
    ax.fill_between(T.flatten(),mean - s, mean + s,alpha=0.5,lw=0)

    ax.plot(T.flatten(),gnd)
    ax.axvline(m.time)
    ax.set_ylabel('$\mathcal{GP}(x)$')
    ax.set_xlabel('time')
    
def plot_hv(model):
    d = model.get_data()
    hv = d.T[-1].flatten()
    fig,ax = plt.subplots()
    ax.plot(hv)

def plot_hv_td(m,T):
    hv = []
    for t in T.flatten():
        hv += [m.get_PCB_hv(time = t)]

    fig,ax = plt.subplots()
    ax.plot(T,hv)
    ax.axvline(m.time)
    ax.set_ylabel('$\mathcal{H}_{PCB}$')
    ax.set_xlabel('time')
    
def plot_ground_truth(bounds,model):
    n = 50
    x = np.linspace(*bounds[0],n)
    xx = np.meshgrid(x,x)
    pts = np.vstack([ele.ravel() for ele in xx]).T

    t = model.time
    
    fig,ax = plt.subplots(2,2)

    X = model.get_data('X')

    vmin = -5
    vmax = 0
    
    for i in [0,1]:
        Fg = []
        F = []
        for pt in pts:
            y = model.GPRs[i].predict_y(np.atleast_2d(np.append(pt,t)))
            F += [y[0].numpy()]
            Fg += [f(pt,t)]
        F = np.array(F).reshape(-1,1)
        Fg = np.array(Fg).reshape(-1,2)

        ax[0,i].pcolor(*xx,Fg.T[i].reshape(n,n), vmin = vmin, vmax = vmax)
        cax = ax[1,i].pcolor(*xx,F.reshape(n,n), vmin = vmin, vmax = vmax)
        
    fig.colorbar(cax)
        
    c = np.arange(len(X))
    for a in ax.flatten():
        a.scatter(*X[:].T,c=c[:],cmap='cool',s = 1)

logging.basicConfig(level=logging.INFO)
bounds = np.array(((-2,2),(-2,2)))
m = pickle.load(open('fast_model_long_obs.p','rb'))

frames = 200

fig, axes = plt.subplots(2,2,sharey= True, sharex = True)
my_ud = MyAnimation(bounds, m, axes.flatten(), frames)
anim = FuncAnimation(fig, my_ud.update, frames = frames, interval = 500, blit = False)
anim.save('test.gif')    
#plt.show()
