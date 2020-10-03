import numpy as np
import time 
import mlrose_hiive as mr
from mrcounter import RHCCounter
from test_harness import *

class LessTraveled(FitnessFunction):
    N = 16
    m1 = 2
    
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


runs = 100

res = {}
suite = TestSuite(0)

#fn = LessTraveled()
#runner = RHCRunner(fn, {'restarts':5, 'max_attempts' : 200})
#res['RHC'] = suite.test(runner, runs)
#print(resource_report(res))

fn = LessTraveled()
runner = SARunner(fn, dict(schedule = mr.GeomDecay(init_temp=20, decay=0.9995, min_temp=0.001),  
                            max_attempts = 200, max_iters = 20000))
res['SA'] = suite.test(runner, runs)
print(pct_time_correct(res, 15))
print(resource_report(res))

fn = LessTraveled()
runner = GARunner(fn, dict(max_attempts=10, pop_size=200,   elite_dreg_ratio=0.9))
res['GA'] = suite.test(runner, runs)

fn = LessTraveled()
runner = MIMICRunner(fn, dict(keep_pct=0.2, pop_size=200, max_attempts=10))
res['MIMIC'] = suite.test(runner, runs)
print(ranks(res))
print(pct_time_correct(res, 15))
print(resource_report(res))
    
        