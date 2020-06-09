import numpy as np
import copy
import logging

def get_expander_points(t,obj):
    #subset of points for expander testing (D \ S)
    subset = obj.D[obj.S[t].mask].reshape(-1,obj.input_dim)            
    #find expander points
    obj.G.append(copy.deepcopy(obj.S[t]))

    #iterate only through the safe set and see if a given point is an "expander"
    for i in range(obj.npts):
        if ~obj.S[t].mask[i]:
            pt = np.atleast_2d(obj.S[t][i])
            
            #a point is an expander if the measurment creates at least one
            # point where its lower bound satisifes all conditions 
                    
            #copy gprc regressors
            temp_gprc = copy.deepcopy(obj.gprc)

            is_expander_test = np.empty((obj.n_cond,len(subset)))                

            #do fake observation and retrain
            for j in range(obj.n_cond):
                temp_mu, temp_std = temp_gprc[j].predict(pt,return_std=True)
                Y_train = np.vstack((obj.C[t-1][:,j].reshape(-1,1),temp_mu + obj.beta*temp_std))
                X_train = np.vstack((obj.X[t-1],pt))
                temp_gprc[j].fit(X_train,Y_train)

                #test if point is expander in all directions
                test_mu,test_std = temp_gprc[j].predict(subset,return_std=True)
                is_expander_test[j] = test_mu.flatten() - obj.beta * test_std > obj.h[j]

            #logging.info(is_expander_test)
            expanded_npts = np.count_nonzero(np.all(is_expander_test,axis=0))
            #logging.info((obj.G[t][i],expanded_npts))
            if not expanded_npts > 0:
                obj.G[t][i] = np.ma.masked

    #logging.info(f'Expander points: {obj.G[t][~obj.G[t].mask]}')
    #logging.info(obj.G[t])
