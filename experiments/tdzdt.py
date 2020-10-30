import numpy as np
import matplotlib.pyplot as plt

class TDZDT:
    def __init__(self, period = 100):
        self.period = period


    def get_PF_input(self, x1, t):
        return [x1,np.abs(np.sin(2.0 * np.pi * t / T))]

    def get_PF_output(self, f1, t):
        x2 = np.abs(np.sin(2.0 * np.pi * t / T))
        return [f1,self.g(x2,t)*(1 - np.sqrt(f1 / self.g(x2,t)))]
    
    def g(self, x, t):
        return 1.0 + np.abs(np.sin(2.0*np.pi*t / T)) + (x[1] - np.abs(np.sin(2.0 * np.pi * t / T)))**2

    def f(self,x,t):
        x1 = x[0]
        x2 = x[1]
        return [x1, self.g(x2,t)*(1 - np.sqrt(x1 / self.g(x2,t)))]
        

def main():
    f1 = np.linspace(0,1)
    T = np.linspace(0.0,1.0,10)

    fig,ax = plt.subplots()
    for t in T:
        f2 = g(t)*(1.0 - np.sqrt(f1 / g(t)))
        print(t)
        ax.plot(f1,f2)

main()
plt.show()
    
