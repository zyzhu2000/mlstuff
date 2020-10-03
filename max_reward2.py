import numpy as np
import time 
import mlrose_hiive as mr
from mrcounter import RHCCounter
from test_harness import *


N = 21
penalty = 1

def count_score(state, val):
    score = 0.0
    p = 0
    for i in range(len(state)):
        if state[i] == val:
            score += 1
            if i==0 or state[i]!=state[i-1]:
                p += penalty
    return score - p

def threholding(s):
    if s<3:
        return 0
    elif 3<=s<16:
        return 50
    else:
        return 100
        

def fn_fitness(state:np.ndarray):
    s0 = count_score(state, 0)
    s1 = count_score(state, 1)
    
    v0 = threholding(s0)
    v1 = threholding(s1)
    
    return v0 + v1

def fn_fitness2(state:np.ndarray):
    s0 = count_score(state, 0)
    s1 = count_score(state, 1)
    
    pv = np.array([s0-15, s1-15, s0-3, s1-3])
    p = scipy.stats.norm.cdf(pv/1)
    
    
    v = (p[0] + p[1])*(1000-100) + (p[2] + p[3])*100
    
    return v 

class MaxReward(FitnessFunction):
    def __call__(self, state):
        self.count += 1
        return fn_fitness2(state)

    def get_problem(self):
        fn = mr.CustomFitness(self)
        problem = mr.DiscreteOpt(length = N, fitness_fn=fn, maximize=True)
        return problem


runs = 100

res = {}
suite = TestSuite(0)

fn = MaxReward()
runner = RHCRunner(fn, {'restarts':5, 'max_attempts' : 200, 'argmax_mode':True})
res['RHC'] = suite.test(runner, runs)
print(resource_report(res))

fn = MaxReward()
runner = SARunner(fn, dict(schedule = mr.GeomDecay(init_temp=10, decay=0.999, min_temp=0.001),  
                            max_attempts = 10, max_iters = 20000))
res['SA'] = suite.test(runner, runs)
print(pct_time_correct(res, 150))
print(resource_report(res))

fn = MaxReward()
runner = GARunner(fn, dict(max_attempts=10, pop_size=200,   elite_dreg_ratio=0.9))
res['GA'] = suite.test(runner, runs)

fn = MaxReward()
runner = MIMICRunner(fn, dict(keep_pct=0.2, pop_size=500, max_attempts=10))
res['MIMIC'] = suite.test(runner, runs)
print(ranks(res))
print(pct_time_correct(res, 15))
print(resource_report(res))
    
