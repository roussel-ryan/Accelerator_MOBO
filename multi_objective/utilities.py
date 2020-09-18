import numpy as np
import pandas as pd


def create_gp_dataframe(self):
    #add data from GPRs
    x_data = self.GPRs[0].data[0].numpy()

    #store the number of observed points in the model
    self.n_observations = len(x_data)

    y_data = np.hstack([ele.data[1].numpy() for ele in self.GPRs])
        
        
    frame_cols = {}
    for i in range(self.input_dim):
        frame_cols[f'X{i}'] = x_data.T[i]

    for j in range(self.obj_dim):
        frame_cols[f'Y{j}'] = y_data.T[j]

    if self._use_constraints:
        c_data = np.hstack([ele.GPR.data[1].numpy() for ele in self.constraints])
        for k in range(self.constr_dim):
            frame_cols[f'C{k}'] = c_data.T[k]
        frame_cols['is_feasable'] = get_feasable_labels(self).astype(bool).tolist()    

    else:
        frame_cols['is_feasable'] = [True] * self.n_observations
            
    frame_cols['in_target_range'] = inside_obj_domain(self,y_data)
                                   
            
    return pd.DataFrame(frame_cols)



def get_feasable_labels(self):
    if self._use_constraints:
        b = []
        for const in self.constraints:
            b += [const.get_feasable()]
            
        b = np.array(b)
        b = np.prod(b,axis=0)
    else:
        b = np.ones(self.n_observations)
            
    return b
        
def get_feasable_idx(self):
    return np.argwhere(self.get_feasable_labels()).flatten()

def inside_obj_domain(self,F):
    return [np.all(ele > self.A) and np.all(ele < self.B) for ele in F]
    

def get_data(self, name = 'all', feas = None, convert = True):
    '''
    get subset of data from dataframe

    name : string, 'X','Y','C', optional (default 'all')
        Specifies which data group to get

    feas : bool, optional (default None)
        Specifies if the points should be feasable or not, defualt gets all points

    '''
    assert name in ['all','X','Y','C']
    

    if feas == None:
        df = self.data
    elif feas:
        df = get_feasable(self)
    else:
        df = get_feasable(self,invert = True)

    if name == 'all':
        pass
    else:
        df = df.filter(regex = f'^{name}',axis=1)

    if convert:
        return df.to_numpy()
    else:
        return df
    
        
def get_feasable(self,invert = False):
    if invert:
        return self.data[~(self.data['is_feasable'] & self.data['in_target_range'])]
    else:
        return self.data[self.data['is_feasable'] & self.data['in_target_range']]
    
