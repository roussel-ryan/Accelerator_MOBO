import numpy as np
import matplotlib.pyplot as plt

import logging

class MultiObjectiveBayesianOptimizer:
    def __init__(self,inputs):
        self.bounds           = inputs.get('bounds',None)
        self.GPRs             = inputs.get('GPRs',None)
        self.A                = inputs.get('A',None)
        self.B                = inputs.get('B',None)

        self.constraints      = inputs.get('constraints',None)

        self.input_dim        = len(self.bounds)
        self.obj_dim          = len(self.GPRs)

        
        
        




        
