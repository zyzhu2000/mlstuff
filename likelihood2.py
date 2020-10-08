import numpy as np, sys
import time, pickle as pkl 
import matplotlib.pyplot as plt
import mlrose_hiive as mr
from mrcounter import RHCCounter
from test_harness import *

N = 64

probs = None


def fn_fitness(state:np.ndarray):
    j = np.arange(len(state)-1)
    idx = np.zeros(len(state), dtype=int)
    idx[1:] = 2*(j+1) + state[:-1] + 1
    P = probs[idx] * (1-state) + (1- probs[idx])*state
    lp =  np.mean(np.log(P)) * 64
    p = np.exp(lp)
    return p*1e18

    
def fn_fitness2(state:np.ndarray):
    v = 1.0
    for i, s in enumerate(state):
        if i==0:
            if s==0:
                v = probs[0]
            else:
                v = 1 - probs[0]
        else:
            pr = probs[1+2*i+p]
            if s==0:
                v *= pr
            else:
                v *= (1-pr)
        p = int(s)
    return v*1e18


class CSequence(FitnessFunction):
    def __call__(self, state):
        self.count += 1
        return fn_fitness(state)

    def get_problem(self):
        fn = mr.CustomFitness(self)
        problem = mr.DiscreteOpt(length = N, fitness_fn=fn, maximize=True)
        return problem

def runner(runs, N_):
    global N, probs
    
    seed = np.random.randint(65565)
    print(seed)
    np.random.seed(seed)
    
    N = N_
    probs = np.random.uniform(0.4, 0.6, 2*N+3)
    
    res = {}
    suite = TestSuite(0)
    
    
    
    fn = CSequence()
    runner = RHCRunner(fn, {'restarts':50,  'argmax_mode':True})
    res['RHC'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    print(pct_time_correct(res, 94))
    print(resource_report(res))
    
    
    fn = CSequence()
    runner = SARunner(fn, dict(schedule = mr.GeomDecay(init_temp=1, decay=0.1),  
                                max_attempts = 8, max_iters = 1000))
    res['SA'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    
    print(pct_time_correct(res, 94))
    print(resource_report(res))
    
    fn = CSequence()
    runner = GARunner(fn, dict(max_attempts=50, pop_size=200, pop_breed_percent=0.6, mutation_prob=0.5,  elite_dreg_ratio=0.9))
    res['GA'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    print(pct_time_correct(res, 94))
    print(resource_report(res))
    
    
    fn = CSequence()
    runner = MIMICRunner(fn, dict(keep_pct=0.3, pop_size=1500, max_attempts=50))
    res['MIMIC'] = suite.test(runner, runs)
    printdf(ranks(res), 'ranks-li')
    printdf(summary_scores(res), "summ-li")
    printdf(pct_time_correct(res, 94), "score-li")
    printdf(resource_report(res), "resource-li")
    
    with open('li-{}.pkl'.format(N), 'wb') as f:
        pkl.dump(res, f)
    

if __name__=='__main__':
    runs = 50
    gs = False
    
    if gs:
        runs = 100
            
        suite = TestSuite(0)
        fn = CSequence()
        runner = SARunner(fn, dict(schedule = mr.GeomDecay(init_temp=2, decay=0.8),  
                                    max_attempts = 20, max_iters = 180))
        
        search_res = grid_search(suite, runner, param_grid=dict(schedule__init_temp=[0.001, 0.01, 0.1, 1, 10, 100], 
                                                                schedule__decay=[0.9999, 0.9999, 0.999, 0.99, 0.9, 0.8, 0.5]), runs=runs)
        s = sorted(search_res.items(), key=lambda x: x[1], reverse=True)
        for line in s:
            print(line)
    
    ###
    
    if len(sys.argv)==1:
        runner(runs, N)
    else:
        prob = [int(x) for x in sys.argv[1:]]
        for i in prob:
            runner(runs, i)
        
    
        
        
