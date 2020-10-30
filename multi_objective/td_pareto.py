import numpy as np
from . import pareto

def get_PCB_PF(model, **kwargs):
    t = kwargs.get('time', model.time)
    gamma = kwargs.get('gamma',1.0)
    
    X = np.hstack([model.get_data('X',time = t), model.get_data('t',time = t)])

    if len(X) == 0:
        return None

    else:
        X[:,-1] = np.ones_like(X[:,-1]) * t
                
        #get prediction from GPs
        PCB = []
        for i in range(model.obj_dim):
            m, s = model.GPRs[i].predict_y(X)
            PCB += [(m + np.sqrt(gamma * s)).numpy()]

        PCB = np.hstack(PCB)

        
        
        return pareto.get_PF(PCB, model.B,low_ref = model.A)
