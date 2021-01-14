import numpy as np
from scipy.spatial import distance

import gpflow
import tensorflow as tf

#from GPy.kern.src.kern import Kern
#from GPy.core import Param

class AdvancedRBF(gpflow.kernels.Kernel):
    """ Advanced RBF GPFlow kernel

    This class extends the GPFlow Kernel class to allow explicit correlations
    in the precision matrix for the RBF kernel.

    This kernel has three 'modes' which specify the type of precision matrix
    for the RBF kernel. This also determines the input shape of the kernel variable S.
    We replace the precision matrix in the RBF kernel

    k(x,x') = exp(- 0.5 * (x - x').T * \Sigma * (x - x'))

    with the decomposition \Sigma = L * L.T, where L is an upper-triangular matrix 
    to make optimization and calculation of the Maharanobis distance easier

    k(x,x') = exp(- 0.5 * (x - x').T * L * L.T * (x - x')) 

    "isotropic"   - L = S * np.eye(dim)
                     S = tf.tensor, shape (1,), 
                     corresponding to 1 / sqrt(lengthscale)
    
    "anisotropic" - L = np.diag(S)
                     S = tf.tensor, shape (dim,),
                     corresponding to 1 / sqrt(lengthscale_i) i E [0,dim]

    "correlated"  - L = upper_tri(S)
                     S = tf.tensor, shape ((dim**2)/2 - D,),
                     corrsponds to \Sigma = L * L.T = 1 / covarience matrix
                     
    

    Attributes
    ----------
    S : gpflow.Parameter, shape (see above)
        Parameter controlling precision matrix
    
    variance : gpflow.Parameter, shape(1,)
        Parameter controlling kernel variance

    mode : string
        Specifies type of precision matrix calculation (see options above)

    """

    def __init__(self,active_dims = None,
                 variance = 1.0, S = np.empty(1),
                 mode = 'correlated',name = 'advancedRBF'):

        
        super(AdvancedRBF, self).__init__(active_dims=None, name = name)

        self.S           = gpflow.Parameter(S)
        self.variance    = gpflow.Parameter(variance)

        self._dtype      = tf.float64
        self.mode        = mode


    def get_precision_matrix(self):
        return tf.linalg.matmul(self.L,tf.transpose(self.L))

    
    def get_covariance_matrix(self):
        return tf.linalg.inv(self.get_precision_matrix())

    
    def K(self, X, X2 = None):
        #get input dimention from input variable
        self.D = tf.shape(X)[1].numpy()
        S = self.S
    
        if self.mode == 'correlated':
            #if using the correlated matrix the input shape should be (D**2 + D)/2
            assert tf.shape(S)[0].numpy() == int(self.D**2 + self.D)/2

            self.L = self._construct_upper_triangle(S)
            
            
        elif self.mode == 'anisotropic':
            #if using anisotropic, L is elements along the diagonal
            assert tf.shape(S)[0].numpy() == self.D

            #need to transform into upper triangle (just diagonal matrix)
            self.L = tf.linalg.diag(S)
            
        elif self.mode == 'isotropic':
            assert tf.shape(L)[0].numpy() == 1

            self.L = tf.linalg.diag(tf.ones(self.D) * S)

        else:
            raise RuntimeError(f'RBF mode {self.mode} not found!')

        
        if X2 is None:
            X2 = X
            dists = self._EfficientMaharanobis(X,X2,self.upper_triangle)
            
            #to make sure that no nans occur,
            #if we try to get K(X,X) set the distance to zero
            dists = tf.linalg.set_diag(dists,tf.zeros(tf.shape(X)[0],dtype=self._dtype))

        else:    
            dists = self._EfficientMaharanobis(X,X2,self.upper_triangle)

        return self.variance * tf.exp(-0.5 * dists ** 2)

    
    def _construct_upper_triangle(self,S):
        #reshape the array L into a lower triangular matrix which guarentees that it is PSD
        #make sure the input array length is (D**2 + D)/2
        D = self.D

        tiu_idx = np.triu_indices(D, k = 0)
               
        mask = np.zeros((D,D),dtype=bool)
        mask[tiu_idx] = True

        idx = np.zeros((D,D),dtype=int)
        idx[tiu_idx] = np.arange(np.shape(tiu_idx)[1])

        upper_triangle = tf.where(mask,tf.gather(S,idx),tf.zeros((D,D),dtype=self._dtype))
        return L

    
    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1],tf.squeeze(self.variance))

    
    def _EfficientMaharanobis(self,A,B,S_half):
        '''
        https://fairyonice.github.io/mahalanobis-tf2.html
        A : tensor, N sample1 by N feat
        B : tensor, N sample2 by N feat
        S : tensor, N feat by N feat 
    
        Output:
    
        marahanobis distance of each pair (A[i],B[j]) with inv variance S
    
    
        '''
        #S_half = tf.linalg.cholesky(invS)
        A_star = tf.matmul(A,S_half)
        B_star = tf.matmul(B,S_half)

        res = self._Euclidean(A_star,B_star)
        return(res)
    
    def _Euclidean(self,A,B):
        v = tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1)
        p1 = tf.reshape(tf.reduce_sum(v,axis=1),(-1,1))
        v = tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1])
        p2 = tf.transpose(tf.reshape(tf.reduce_sum(v,axis=1),(-1,1)))
        res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True))
        return(res)

if __name__=='__main__':
    obj = AdvancedRBF()
    x1 = np.array((0,0)).reshape(-1,1)
    x2 = np.array((2,2)).reshape(-1,1)
    print(obj._Euclidean(x1,x2))
