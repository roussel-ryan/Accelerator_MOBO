import numpy as np
import matplotlib.pyplot as plt

import subprocess

def generate_input(x,P,r):
    '''
    generate input file for C++ program by Emmermich only works for 3D!
    x - evaluation points w/ [*<mean>,*<std>]
    '''

    n = len(P)

    fname = 'input.txt'
    with open(fname,'w') as f:
        f.write(f'{n}\n')

        #write observed points (dominated or non-dominated is ok
        for pt in P:
            s = ' '.join(pt.astype(str)) + '\n'
            f.write(s)

        #write reference point
        f.write('\n')
        f.write(' '.join(r.astype(str)) + '\n')

        #write evaluation point(s)
        for pt in x:
            f.write(' '.join(pt.astype(str)) + '\n')
    return fname
    
def run_script(infile):
    p = subprocess.run(['EHVI.exe',infile,'sliceupdate'],stdout=subprocess.PIPE)
    return np.asfarray(p.stdout.split())
    
if __name__=='__main__':
    #test input generation and run_script
    x = np.random.uniform(size = (2,6))
    P = np.random.uniform(size = (3,3))
    r = np.array((2.0,2.0,2.0))
    f = generate_input(x,P,r)
    run_script(f)
        
            
    
    
    
