import numpy as np
import matplotlib.pyplot as plt
import GPy


X = np.random.uniform(-3,3,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05

kernel = GPy.kern.RBF(input_dim = 1, variance = 0.25,lengthscale = 1.)

m = GPy.models.GPRegression(X,Y,kernel)
print(help(m.constrain_bounded))
m.kern.lengthscale.constrain_bounded(0.0,0.5)
m.optimize()

fig = m.plot(plot_density=True)
#fig,ax = plt.subplots()
#ax.plot(x,mu[0])
#ax.fill_between(x.flatten(),(mu[0] - mu[1]).flatten(),(mu[0] + mu[1]).flatten(),alpha=0.5)

print(m)
plt.show()
