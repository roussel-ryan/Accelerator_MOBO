import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

#suppress output messages below "ERROR" from tensorflow
#and prevent the use of any system GPU's
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import gpflow

from Accelerator_MOBO.kernels import advanced_RBF


def f(x):
    f1 = tf.linalg.norm(x - tf.ones(2))
    return tf.reshape(f1,(1,))


def main():
    '''
    demonstrate use of physics informed kernel for GP
    '''

    X0 = tf.random.uniform((5,2))
    Y0 = tf.stack([f(ele) for ele in X0])

    X0 = tf.cast(X0,tf.float64)
    Y0 = tf.cast(Y0,tf.float64)
    
    #create kernels
    #by default the kernel coefficient is 1
    #isotropic - value is inverse sqrt(lengthscale)
    S = tf.ones(1, dtype = tf.float64)
    k1 = advanced_RBF.AdvancedRBF(S = S, mode = 'isotropic', input_dim = 2)

    #anisotropic
    #elements are inverse sqrt(lengthscales)
    S = tf.constant((1.0,0.5))
    k2 = advanced_RBF.AdvancedRBF(S = S, mode = 'anisotropic', input_dim = 2)

    #correlated - allows retraining of matrix elements - elements are upper triangle elements of L
    #precision matrix = L * L^T
    S = tf.constant((1.0, 0.5, 1.0))
    k3 = advanced_RBF.AdvancedRBF(S = S, mode = 'correlated', input_dim = 2)

    #physics informed - does NOT allow retraining due to symmetry/invertability requirements
    #S is precision matrix
    S = tf.constant(((1.0,0.5),(0.5,1.0)))
    k4 = advanced_RBF.AdvancedRBF(S = S, mode = 'physics', input_dim = 2)


    #for each kernel plot the 2d predictions of the mean
    n = 20
    x = np.linspace(-2,2,n)
    XX = np.meshgrid(x,x)
    pts = np.vstack([ele.ravel() for ele in XX]).T
    pts = tf.convert_to_tensor(pts)

    for k in [k1,k2,k3,k4]:
        print(f'doing {k.mode} kernel\n------------------')
        gpr = gpflow.models.GPR((X0,Y0), k, noise_variance = 0.001)

        print(f'Precision matrix\n {k.get_precision_matrix()}')
        print(f'Covariance matrix\n {k.get_covariance_matrix()}')
        print('\n')
        F = gpr.predict_y(pts)

        fig,ax = plt.subplots()
        ax.pcolor(*XX, tf.reshape(F[0],(n,n)))
        ax.plot(*X0.numpy().T,'+C1')
        
main()
plt.show()
