import matplotlib.pyplot as plt

def plot_acq(self, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        
    self.PF = self.get_PF()
    fargs = [self.GPRs,self.PF,self.A,self.B]    

    n = 30
    x = np.linspace(*self.bounds[0,:],n)
    y = np.linspace(*self.bounds[1,:],n)
    xx, yy = np.meshgrid(x,y)
    pts = np.vstack((xx.ravel(),yy.ravel())).T

    f = []
    for pt in pts:
        f += [self.obj(pt,*fargs)]

    f = np.array(f).reshape(n,n)
    
    c = ax.pcolor(xx,yy,f)
    ax.figure.colorbar(c,ax=ax)          
    
    ax.plot(*self.GPRs[0].data[0].numpy().T,'+')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
        
    return ax

def plot_constr(self, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        
    fargs = [self.GPRs,self.PF,self.A,self.B]    

    n = 30
    x = np.linspace(*self.bounds[0,:],n)
    y = np.linspace(*self.bounds[1,:],n)
    xx, yy = np.meshgrid(x,y)
    pts = np.vstack((xx.ravel(),yy.ravel())).T
    
    f = []
    for pt in pts:
        f += [self.constraints[0].predict(np.atleast_2d(pt))]

    f = np.array(f).reshape(n,n)
    
    c = ax.pcolor(xx,yy,f)
    ax.figure.colorbar(c,ax=ax)          

    X_feas = self.get_feasable_X()
    X_nonfeas = self.get_feasable_X(invert = True)
        
    ax.plot(*X_feas.T,'+r')
    ax.plot(*X_nonfeas.T, 'or')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
        
    return ax
