import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def main():
    bounds = np.array(((-2,2),(-2,2)))
    #opt(bounds)
    m = pickle.load(open('fast_model_long_obs.p','rb'))
    #plot_ground_truth(bounds, m)

    data = m.data.to_numpy()
    #plot_objective_space_density(m,m.time)
    #plot_objective_space_density(m,m.time + 20)
    #plot_objective_space_density(m,75.0)
    t = np.linspace(0,m.time).reshape(-1,1)

    #test = np.array((np.cos(np.pi/4),np.sin(np.pi/4))) * 0.5
    test = np.zeros(2)
    plot_time_dep(m,1,test,t)
    plot_hv_td(m,t)
    
    #m.time = m.time - 10
    #vis(bounds,m)
    #m.time = m.time + 20
    #vis(bounds,m)

    #m.time = m.time + 140
    #vis(bounds,m)



class Problem:
    def __init__(self,func):
        self.func = func
        self.t = 0

    def measure(self,X):
        x = np.array((*(X.flatten()),self.t)).reshape(-1,3)
        y = self.func(X,self.t).reshape(-1,2)
        self.t += 10.0
        return x,y
        
def f(x,t):
    T = 200
    theta = 2 * np.pi * t / T + np.pi/4
    x01 = np.array((np.cos(theta),np.sin(theta))) * 0.5
    x02 = np.array((np.cos(theta),np.sin(theta))) * -0.5
    #x01 = np.array((1,1)) * 0.5 * np.cos(np.pi/4)
    #x02 = np.array((-1,-1)) * 0.5 * np.cos(np.pi/4)

    f1 = np.linalg.norm(x - x01) - 5.0
    f2 = np.linalg.norm(x - x02) - 5.0
    
    return np.array([f1, f2])


def opt(bounds):
    '''Time dependant SOBO with GaussianProcessTools
    
    Each observation costs 1.0 time step so we have a lengthscale ~1.0


    '''
    #start the clock
    t = 0
    toy = Problem(f)
    
    #sample the objective functions over 
    n_initial = 10
    X0 = np.random.uniform(*bounds[0],size = (n_initial,2))
    X = []
    Y = []
    for i in range(n_initial):
        x, y = toy.measure(X0[i])
        X += [x]
        Y += [y]

    X = np.vstack(X)
    Y = np.vstack(Y)
    X[:,-1] = np.arange(n_initial)

    
    #define kernels to be used for the gaussian process regressors
    kernels = [gpflow.kernels.RBF(lengthscales = [0.4, 0.4, 25.0], variance = 1.0) for i in [0,1]]
    mean_funcs = [gpflow.mean_functions.Constant(0.0) for i in [0,1]]
    
    
    B = np.zeros(2)
    A = np.ones(2)*-5.0
    
    #define GP models
    GPRs = []
    for i in [0,1]:
        gpr = gpflow.models.GPR((X,Y[:,i].reshape(-1,1)), kernels[i],
                                noise_variance = 2e-6,
                                mean_function = mean_funcs[i])
        gpflow.set_trainable(gpr.likelihood.variance,False) 
        #gpflow.set_trainable(gpr.kernel.lengthscales,False) 
        GPRs += [gpr]

    opt = evolutionary.SwarmOpt(generations = 10,
                                population = 16)

    acq = infill.TDACQ(infill.NUHVI(beta = 2.0, gamma = 1.0))

    
    mobo_opt = mobo.TDMultiObjectiveBayesianOptimizer(bounds,
                                                      GPRs, B, A = A,
                                                      acq = acq,
                                                      optimizer = opt)

    #sobo_opt.train(1000)

    init_time = X[-1,-1]
    time_steps = 400
    for t in range(time_steps):
        #find next point for observation
        mobo_opt.time = init_time + t * 1.0
        result = mobo_opt.get_next_point()
        X_new = np.atleast_2d(result.x)
        Y_new = f(X_new,mobo_opt.time).reshape(-1,2)
        t_new = np.array((mobo_opt.time)).reshape(1,1)

        print(mobo_opt.data)
        hv = np.array((mobo_opt.get_PCB_hv())).reshape(1,1)
        
        if t % 10 == 0:
            mobo_opt.train(5000)
            mobo_opt.save('fast_model_long_obs.p')
    
        #add observations to mobo GPRs
        
        mobo_opt.add_observations(X_new,Y_new,{'t':t_new,'hv':hv})

    mobo_opt.save('fast_model_long_obs.p')
        
    mobo_opt.print_model()
    
    


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

    
if __name__=='__main__':
    main()
    plt.show()
