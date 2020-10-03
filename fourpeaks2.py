import numpy as np
import time, pickle as pkl 
import matplotlib.pyplot as plt
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



if __name__=='__main__':
    
    runs = 50
    
    res = {}
    suite = TestSuite(0)
    
    fn = FourPeaks()
    runner = RHCRunner(fn, {'restarts':10,  'argmax_mode':True})
    res['RHC'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    print(pct_time_correct(res, 94))
    print(resource_report(res))
    
    
    fn = FourPeaks()
    runner = SARunner(fn, dict(schedule = mr.GeomDecay(init_temp=2, decay=0.8),  
                                max_attempts = 8, max_iters = 1000))
    res['SA'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    
    print(pct_time_correct(res, 94))
    print(resource_report(res))
    
    fn = FourPeaks()
    runner = GARunner(fn, dict(max_attempts=50, pop_size=200, pop_breed_percent=0.75, mutation_prob=0.6,  elite_dreg_ratio=0.9))
    res['GA'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    print(pct_time_correct(res, 94))
    print(resource_report(res))
    
    
    fn = FourPeaks()
    runner = MIMICRunner(fn, dict(keep_pct=0.1, pop_size=1500, max_attempts=50))
    res['MIMIC'] = suite.test(runner, runs)
    printdf(ranks(res), 'ranks')
    printdf(summary_scores(res), "summ")
    printdf(pct_time_correct(res, 94), "score")
    printdf(resource_report(res), "resource")
    
    with open('4peaks.pkl', 'wb') as f:
        pkl.dump(res, f)
        
        
