import numpy as np
import time 
import mlrose_hiive as mr
from mrcounter import RHCCounter
from test_harness import *

class LessTraveled(FitnessFunction):
    N = 17
    m1 = 4
    
    def __call__(self, state):
        self.count += 1
        
        state = state.astype(int)
        x1 = state[0:self.m1]
        x2 = state[self.m1:]
        
        
        if all(x1==0):
            v = np.sum(1-x2) + 1
        else:
            v = np.sum(x2)
        
        return float(v)

    def get_problem(self):
        fn = mr.CustomFitness(self)
        problem = mr.DiscreteOpt(length = self.N, fitness_fn=fn, maximize=True)
        return problem
    
