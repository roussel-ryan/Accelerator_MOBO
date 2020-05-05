import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

from scipy.stats import norm
from scipy.optimize import minimize as sci_minimize 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import logging
logging.basicConfig(level=logging.INFO)

import pareto_tools as PT
import EI_tools as EIT

from ocelot_minimize import minimize as oce_minimize
from GaussianProcessTools.simple_opt import minimize as gp_minimize

def layered_minimization(func,bounds,n_restarts = 10, args=()):
    min_val = 10000000
    dim = len(bounds)
    nfev = 0

    s = time.time()
    for x0 in np.random.uniform(bounds[:,0],bounds[:,1], size = (n_restarts,dim)):
        res = sci_minimize(func, x0 = x0, args = args,bounds = bounds, method='L-BFGS-B')
        nfev = nfev + res.nfev
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
    logging.info(f'number of function evaluations {nfev}, avg exec time {(time.time() - s)/nfev}')
    return min_x

def get_meshes(gps,train_data,r,lim):
    x = train_data.T[:2]

    P = train_data.T[2:].T
    s = PT.get_non_dominated_set(P)
    s = PT.sort_along_first_axis(s)[::-1]
    
    #plot the EHVI as a function of (x1,x2)
    t = 20
    x1 = np.linspace(*lim,t)
    x2 = np.linspace(*lim,t)
    xx,yy = np.meshgrid(x1,x2)
    pts = np.vstack((xx.ravel(),yy.ravel())).T
    ehvi = np.zeros_like(xx.ravel())
    
    f1,f1_std = gps[0].predict(pts,return_std=True)
    f2,f2_std = gps[1].predict(pts,return_std=True)

    for i in range(t**2):
        ehvi[i] = EHVI(np.array((f1[i],f2[i])),np.array((f1_std[i],f2_std[i])),s,r)

    return xx,yy,f1.reshape(t,t),f2.reshape(t,t),ehvi.reshape(t,t)

def test_function(x,y):
    f1 = np.sqrt((x - 1)**2 + (y - 1)**2)
    f2 = np.sqrt((x + 1)**2 + (y + 1)**2)
    #f1 = 4*x**2 + 4*y**2
    #f2 = (x-5)**2 + (y-5)**2
    return np.array((f1,f2))

#function used in scipy.minimize
def get_EHVI(x,GPRs,s,r):
    '''
    x: point input from optimizer
    s: non-dominated set of points
    GPRs: list of GP regressors
    r: reference point
    '''
    #logging.info(f'calling get_EHVI() with point:{x.reshape(-1,2)}')
    f = np.array([ele.predict(x.reshape(-1,2),return_std=True) for ele in GPRs]).T[0]
    #logging.info(f)
    return -EHVI(f[0],f[1],s,r)
    
    
def plot_EHIV(train_data,r,gp1,gp2,lim):
    x = train_data.T[:2]

    P = train_data.T[2:].T
    s = PT.get_non_dominated_set(P)
    s = PT.sort_along_first_axis(s)[::-1]
    
    #plot the EHVI as a function of (x1,x2)
    t = 10
    x1 = np.linspace(*lim,t)
    x2 = np.linspace(*lim,t)
    xx,yy = np.meshgrid(x1,x2)
    pts = np.vstack((xx.ravel(),yy.ravel())).T
    ehvi = np.zeros_like(xx.ravel())
    
    f1,f1_std = gp1.predict(pts,return_std=True)
    f2,f2_std = gp2.predict(pts,return_std=True)

    fig,ax = plt.subplots(3,1,sharex=True,sharey=True)
    ax[0].pcolor(xx,yy,f1.reshape(t,t))
    ax[1].pcolor(xx,yy,f2.reshape(t,t))
    for i in [0,1]:
        ax[i].plot(*x,'r+')
        ax[i].plot((-1,1),(-1,1))
    
    for i in range(t**2):
        ehvi[i] = EHVI(np.array((f1[i],f2[i])),np.array((f1_std[i],f2_std[i])),s,r)

    ax[2].pcolor(xx,yy,ehvi.reshape(t,t))
    for i in range(3):
        ax[i].set_ylabel('x2')
        ax[i].set_xlabel('x1')
    
    fig2,ax2 = plt.subplots()
    ax2.plot(P.T[0],P.T[1],'+')
    ax2.plot((0,2*np.sqrt(2)),(2*np.sqrt(2),0))
    ax2.set_xlim(0,7)
    ax2.set_ylim(0,7)
    ax2.set_ylabel('f2')
    ax2.set_xlabel('f1')
    
    
def show_EHVI(s):
    fig,ax = plt.subplots()
    s_min = np.min(s.T,axis=1)
    s_max = np.max(s.T,axis=1)
    #logging.info(s_min)
    l = 50
    x_test = np.linspace(0.8*s_min[0],1.2*s_max[0],l)
    y_test = np.linspace(0.8*s_min[1],1.2*s_max[1],l)

    xxt,yyt = np.meshgrid(x_test,y_test)
    pts = np.vstack((xxt.ravel(),yyt.ravel())).T

    sigma = np.array((0.0,0.0))
    r = np.array((1000,1000))
    
    ehvi = np.zeros_like(xxt.ravel())
    for i in range(len(xxt.ravel())):
        ehvi[i] = EHVI(pts[i],sigma,s,r)

    m = ax.pcolor(xxt.reshape(l,l),yyt.reshape(l,l),ehvi.reshape(l,l),alpha=0.5)
    ax.plot(s.T[0],s.T[1])
    
    fig.colorbar(m)
    
def EHVI(mu,sigma,Y,r):
    #number of non-dominated points
    n = len(Y)
    
    #add bounding points to set
    Y = np.vstack(((r[0], -np.inf),Y,(-np.inf,r[1])))
    #logging.info(Y)

    sum1 = 0
    sum2 = 0
    for i in range(1,n+1):
        sum1 = sum1 + (Y[i-1][0] - Y[i][0])*EIT.CDF(Y[i][0],mu[0],sigma[0])*EIT.PSI(Y[i][1],Y[i][1],mu[1],sigma[1])
        sum2 = sum2 + (EIT.PSI(Y[i-1][0],Y[i-1][0],mu[0],sigma[0]) - EIT.PSI(Y[i-1][0],Y[i][0],mu[0],sigma[0])) * EIT.PSI(Y[i][1], Y[i][1], mu[1],sigma[1]) 
        #logging.info(f'sum terms for rect {i}: {sum1},{sum2}')
        #logging.info(f'PSI(Y[{i}][1], Y[{i}][1], mu[1], sigma[1]): {EIT.PSI(Y[i][1], Y[i][1], mu[1],sigma[1])}')
        #logging.info(f'EIT.PSI(Y[{i-1}][0],Y[{i-1}][0],mu[0],sigma[0]): {EIT.PSI(Y[i-1][0],Y[i-1][0],mu[0],sigma[0])}')
        #logging.info(f'EIT.PSI(Y[{i-1}][0],Y[{i}][0],mu[0],sigma[0]): {EIT.PSI(Y[i-1][0],Y[i][0],mu[0],sigma[0])}')
    return sum1 + sum2

##################################################################################
##################################################################################



#np.random.seed(1)
n = 3
lim = [-2,2]
x = np.random.uniform(*lim,n)
y = np.random.uniform(*lim,n)
#x = np.random.randn(n)*5.0
#y = np.random.randn(n)*3.0 
train_data = np.empty((n,4))

for i in range(n):
    f = test_function(x[i],y[i])
    train_data[i] = np.array((x[i],y[i],f[0],f[1]))


kernel = RBF()#length_scale = 0.5)
gp1 = GaussianProcessRegressor(kernel = kernel, alpha = 0.01,n_restarts_optimizer=5)
gp2 = GaussianProcessRegressor(kernel = kernel, alpha = 0.0001)

    
    #train models
gp1.fit(train_data.T[:2].T,train_data.T[2])
gp2.fit(train_data.T[:2].T,train_data.T[3])
GPRs = [gp1,gp2]
#plot 2d front
#ax.plot(train_data.T[2],train_data.T[3],'+')

x = train_data.T[:2]
P = train_data.T[2:].T
#logging.info(P)
S = PT.get_non_dominated_set(P)

#sort points in decreasing order of f1
S = PT.sort_along_first_axis(S)[::-1]
r = np.array((10,10))

#optimization
x0 = np.array((0.0,0.0))

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.set_ylabel('x2')
ax1.set_title('GP - f1')
ax2.set_title('GP - f2')
ax3.set_ylabel('x2')
ax3.set_xlabel('x1')
ax4.set_xlabel('f1')
ax4.set_ylabel('f2')

#create axes artists
xx,yy,f1,f2,ehvi = get_meshes(GPRs,train_data,r,lim)

#gp1
in_data1,= ax1.plot(*x,'r+')
gp_mesh1 = ax1.pcolor(xx,yy,f1)

#gp2
in_data2, = ax2.plot(*x,'r+')
gp_mesh2 = ax2.pcolor(xx,yy,f2)

#ehvi
m = np.max(ehvi)
ehvi = ehvi / m
ehvi_mesh = ax3.pcolor(xx,yy,ehvi)
in_data3, = ax3.plot(*x,'r+')
m_text = ax3.text(0.01,0.99,f'Max EHVI: {m:.2f}',transform=ax3.transAxes,ha='left',va='top',c='w')

#objectives
obj, = ax4.plot(*P.T,'r+')
best, = ax4.plot((np.sqrt(2)*2,0),(0,np.sqrt(2)*2))
ax4.set_xlim(0,4)
ax4.set_ylim(0,4)



def update():
    global S
    global train_data
    #do minimization
    #res = sci_minimize(get_EHVI,x0,args = (GPRs,S,r),bounds = (lim,lim),tol=1e-3,method='TNC')
    #new_pt = res.x
    bounds = np.array((lim,lim))
    new_pt = layered_minimization(get_EHVI,bounds,args = (GPRs,S,r))
    #new_point, _ = minimize(np.array((0,0)),get_EHVI,args = ( GPRs,S,r),maxnumlinesearch = 10)
    
    #new_pt, _ = gp_minimize(get_EHVI,bounds,args = (GPRs,S,r))

    #find function value at new point (observation)
    f = test_function(*new_pt)
        
    #append new observation to dataset
    train_data = np.vstack((train_data,np.array((*new_pt,*f))))

    #update GP models
    for j in range(len(GPRs)):
        GPRs[j].fit(train_data.T[:2].T,train_data.T[2+j])

    #update pareto front
    P = train_data.T[2:].T
    S = PT.get_non_dominated_set(P)
    S = PT.sort_along_first_axis(S)[::-1]

    
    

def animate(i):
    logging.info(f'making animation frame {i}')
    update()
    #update plots
    xx, yy,f1,f2,ehvi = get_meshes(GPRs,train_data,r,lim)
    x = train_data.T[:2]
    P = train_data.T[2:]

    m = np.max(ehvi)

    #gp1
    in_data1.set_data(*x)
    f1 = f1[:-1,:-1]
    gp_mesh1.set_array(f1.ravel())
    #gp2
    in_data2.set_data(*x)
    f2 = f2[:-1,:-1]
    gp_mesh2.set_array(f2.ravel())
    #ehvi
    in_data3.set_data(*x)
    ehvi = ehvi[:-1,:-1] / m
    ehvi_mesh.set_array(ehvi.ravel())
    m_text.set_text(f'Max EHVI: {m:.2f}') 
    #obj
    obj.set_data(*P)

    return in_data1,gp_mesh1,in_data2,gp_mesh2,ehvi_mesh,obj


logging.info('making animation')
anim = animation.FuncAnimation(fig,animate,frames = 50,interval = 1,blit=False,repeat=False)
#plt.show()
print(animation.writers.list())
anim.save('movie.gif',writer='pillow',dpi=300,fps = 1)
logging.info(GPRs[1].kernel_.get_params())
    
