import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_spherical_mesh(dim,n_r,n_theta,r_max, half_sphere = False):
    
    r = np.linspace(r_max/n_r,r_max,n_r)
    thetas = []
    for i in range(dim-2):
        if half_sphere:
            thetas.append(np.linspace(0.5*np.pi/(n_theta + 1),0.5*np.pi * (1 - 1/(n_theta+1)),n_theta))
        else:
            thetas.append(np.linspace(np.pi/(n_theta + 1),np.pi * (1 - 1/(n_theta+1)),n_theta))

    thetas.append(np.linspace(0,2*np.pi,n_theta+1)[:-1])
    #create mesh
    mesh = np.meshgrid(r,*thetas)
    pts = np.vstack([mesh[i].ravel() for i in range(dim)]).T

    cpts = np.zeros((len(pts),dim))
    for i in range(len(pts)):
        cpts[i] = transform_spherical_to_cartesian(pts[i])

    #add points along z-axis
    axis_pts = np.zeros((n_r,dim))
    axis_pts[:,0] = r
    cpts = np.vstack((axis_pts,cpts))
    axis_pts[:,0] = -r
    cpts = np.vstack((axis_pts,cpts))
    
    
    cpts = np.vstack((np.zeros(dim),cpts))
    print(f'generated spherical mesh with {len(cpts)} points')
    return cpts



def transform_spherical_to_cartesian(pt):
    r = pt[0]
    dim = len(pt)
 
    cpt = np.ones((dim+1)) * r

    #NOTE: INDEXING IS SHIFTED TO MATCH MATH NOTATION,INDEXES START AT 1 
    
    for i in range(1,dim+1):
        #if not first dimension add sin() terms
        if not i == 1:
            for j in range(1,i):
                cpt[i] = cpt[i] * np.sin(pt[j])
                

        #if the last dimension multiply by sin
        if i == dim:
            #finally multiply by sin
            #cpt[i] = cpt[i] * np.sin(pt[i-1])
            pass
        else:
            #otherwise multiply by cos
            cpt[i] = cpt[i] * np.cos(pt[i])

    return cpt[1:]
    
if __name__ == '__main__':
    #pt = np.array((1.0,np.pi/6,np.pi/3,np.pi))

    #print(transform_spherical_to_cartesian(pt))
    mesh = generate_spherical_mesh(5,2,3,1)
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')

    #ax.plot(*mesh.T,'+')
    #plt.show()
    
