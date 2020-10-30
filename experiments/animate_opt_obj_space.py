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
        self.n = 300
        self.x = np.linspace(-5,-3.5,self.n)
        self.xx = np.meshgrid(self.x,self.x)
        self.pts = np.vstack([ele.ravel() for ele in self.xx]).T

        self.t = np.linspace(0, 400, tsteps + 1)
        
        #initialize meshes
        self.quads = []
        for ax in axes:
            self.quads += [ax.pcolor(*self.xx,
                                     np.zeros((self.n,self.n)),
                                     cmap = 'Greys',
                                     vmin = 0,vmax = 8)]

        cbar = axes[0].figure.colorbar(self.quads[0])
        cbar.set_label('log10(P(f,t))')
        #initialize points
        self.PCB_lines = []
        for ax in axes:
            ln, = ax.plot([],[],'+r')
            self.PCB_lines += [ln]

        axes[0].set_ylabel(f'$f_2$')
        axes[0].set_xlabel(f'$f_1$')


            
        #ideal lines
        #self.ideal_lines = []
        #for ax in axes:
        #    ln, = ax.plot([],[],'-C1')
        #    self.ideal_lines += [ln]
        
    def transform_for_pcolor(self,arr):
        return arr.reshape(self.n,self.n)[:-1,:-1].ravel()
            
    def update(self,i):
        logging.info(i)

        #get mu,sigma for each observation point at time t
        t = self.t[i + 1]
        X = self.model.get_data('X',time = t)
        n_pts = len(X)
        X = np.hstack((X, np.ones((n_pts,1))*t))
        #logging.info(X)
        
        pred1 = self.model.GPRs[0].predict_y(X)
        pred2 = self.model.GPRs[1].predict_y(X)

        pred = np.hstack((pred1[0].numpy(),pred2[0].numpy(),
                          pred1[1].numpy(),pred2[1].numpy()))
        
        #logging.info(n_pts)
        #add up the probability contributions
        F = np.zeros(len(self.pts))
        
        for p in pred:
            rv = multivariate_normal(p[:2],np.diag(p[2:]))
            F = F + rv.pdf(self.pts)

        gamma = 1
        PCB = np.vstack((pred[:,0] + gamma * pred[:,2]**0.5, pred[:,1] + gamma * pred[:,3]**0.5)).T
        #print(PCB)
        self.PCB_lines[0].set_data(*PCB.T)
        
        
        F = self.transform_for_pcolor(F)
        F_log = np.log10(F)

        self.quads[0].set_array(F_log)

        #f1 = self.transform_for_pcolor(np.array(f1))
        #f2 = self.transform_for_pcolor(np.array(f2))
        #gp_f1 = self.transform_for_pcolor(np.array(gp_f1))
        #gp_f2 = self.transform_for_pcolor(np.array(gp_f2))
        
        
        

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

    
    
    cax = fig.colorbar(c)
    cax.set_label('$\log_{10}(P(f_1,f_2,t))$')

    
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
    

logging.basicConfig(level=logging.INFO)
bounds = np.array(((-2,2),(-2,2)))
m = pickle.load(open('fast_model_long_obs.p','rb'))

frames = 200

fig, axes = plt.subplots(1,1,sharey= True, sharex = True)
my_ud = MyAnimation(bounds, m, [axes], frames)
anim = FuncAnimation(fig, my_ud.update, frames = frames, interval = 500, blit = False)
anim.save('obj_test.gif')    
#plt.show()
