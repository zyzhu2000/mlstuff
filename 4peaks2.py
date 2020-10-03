import numpy as np
import time 
import mlrose_hiive as mr
from mrcounter import RHCCounter
from test_harness import *



N = 50
T = 5

def head(state, val):
    idx = np.nonzero(state==val^1)[0]
    if len(idx)==0:
        head1 = len(state)
    else:
        head1 = idx[0] 
    return head1

def tail(state, val):
    return head(state[::-1], val)

def fn_fitness(state:np.ndarray):
    head1 = head(state, 1)
    tail0 = tail(state, 0)
    
    reward = 0
    if tail0 > T and head1 >T:
        reward = N
    v = max(tail0, head1)
    return reward + v



class FourPeaks(FitnessFunction):
    def __call__(self, state):
        self.count += 1
        return fn_fitness(state)

    def get_problem(self):
        fn = mr.CustomFitness(self)
        problem = mr.DiscreteOpt(length = N, fitness_fn=fn, maximize=True)
        return problem


runs = 5

res = {}
suite = TestSuite(0)

fn = FourPeaks()
runner = RHCRunner(fn, {'restarts':5, 'max_attempts' : 200, 'argmax_mode':True})
res['RHC'] = suite.test(runner, runs)
print(ranks(res))
print(summary_scores(res))
print(pct_time_correct(res, 15))
print(resource_report(res))


fn = FourPeaks()
runner = SARunner(fn, dict(schedule = mr.ExpDecay(exp_const=0.001),  
                            max_attempts = 1, max_iters = 1000))
res['SA'] = suite.test(runner, runs)
print(ranks(res))
print(summary_scores(res))

print(pct_time_correct(res, 15))
print(resource_report(res))

fn = FourPeaks()
runner = GARunner(fn, dict(max_attempts=50, pop_size=200,   elite_dreg_ratio=0.9))
res['GA'] = suite.test(runner, runs)
print(ranks(res))
print(summary_scores(res))
print(pct_time_correct(res, 15))
print(resource_report(res))


fn = FourPeaks()
runner = MIMICRunner(fn, dict(keep_pct=0.2, pop_size=2000, max_attempts=50))
res['MIMIC'] = suite.test(runner, runs)
print(ranks(res))
print(summary_scores(res))
print(pct_time_correct(res, 15))
print(resource_report(res))
    
