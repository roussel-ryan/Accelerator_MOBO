import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

def f(X):
    return np.linalg.norm(X)

#plot 2d kernel where x0 = (0,0)
n = 100
x0 = np.zeros((1,2))
x = np.linspace(-1,1,n)
xx, yy = np.meshgrid(x,x)
pts = np.vstack((xx.ravel(),yy.ravel())).T

X = np.random.uniform(-1,1,(20,2)).reshape(-1,2)
Y = np.array([f(x) for x in X]).reshape(-1,1)

k = gpflow.kernels.RBF(lengthscales = np.array([[1.0,2.0],[0.1,1.0]]))

m = gpflow.models.GPR(data = (X,Y),kernel=k,mean_function=None)

opt = gpflow.optimizers.Scipy()
#opt = tf.optimizers.Adam() 
print('doing minimization')
opt.minimize(m.training_loss,m.trainable_variables)
print_summary(m)

fig,ax = plt.subplots()
ax.pcolor(xx,yy,m.predict_f(pts)[0].numpy().reshape(n,n))

plt.show()
