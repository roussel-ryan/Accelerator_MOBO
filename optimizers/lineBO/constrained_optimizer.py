import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.optimize as opt

from . import optimizer as lineOpt
from ..stageopt import optimizer as stageOpt

class StageLineOpt(lineOpt.LineOpt):
    def __init__(self,bounds,acq,**kwargs):
        

        super().__init__(bounds,acq,**kwargs)
