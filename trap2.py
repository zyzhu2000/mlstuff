import numpy as np
import time, pickle as pkl 
import matplotlib.pyplot as plt
import mlrose_hiive as mr
from mrcounter import RHCCounter
from test_harness import *

N = 30

    
def fn_fitness(state:np.ndarray):
    u = np.sum(state)
    a = 70
    b = 100
    z = 18*N//30
    
    if u<=z:
        v = a/z*(z-u)
    else:
        v = b/(N-z)*(u-z)
        
    return v 


class TrapDiscreteOpt(mr.DiscreteOpt):
    def __init__(self, length, fitness_fn, maximize):
        super().__init__(length = length, fitness_fn=fitness_fn, maximize=maximize)
        
    def random_neighbor(self):
        # equalize the probability of 0->1 and 1->0
        neighbor = np.copy(self.state)
        if np.random.uniform()<0.042:
            for j in range(self.length):
                neighbor[j] ^= 1      
            return neighbor
        
        idx1 = np.nonzero(self.state==1)[0]
        if len(idx1)==0:
            idx = np.nonzero(self.state==0)[0]
        elif len(idx1)==self.length:
            idx = idx1
        elif np.random.uniform()>0.5:
            idx = np.nonzero(self.state==0)[0]
        else:
            idx = idx1
            
            
        neighbor = np.copy(self.state)    
        i = np.random.randint(0, len(idx))    
        neighbor[idx[i]] ^= 1
        return neighbor
    
    def random_neighbor2(self):
        neighbor = np.copy(self.state)
        
        if np.random.uniform()<0.2:
            for j in range(self.length):
                neighbor[j] ^= 1
        else:
            i = np.random.randint(0, self.length)
            neighbor[i] ^= 1

        return neighbor


class Trap(FitnessFunction):
    def __call__(self, state):
        self.count += 1
        return fn_fitness(state)

    def get_problem(self):
        fn = mr.CustomFitness(self)
        #problem = mr.DiscreteOpt(length = N, fitness_fn=fn, maximize=True)
        problem = TrapDiscreteOpt(length = N, fitness_fn=fn, maximize=True)
        return problem

def runner(runs, N_):
    global N
    N = N_
    res = {}

    suite = TestSuite(0)
    
    fn = Trap()
    runner = RHCRunner(fn, {'restarts':20,  'argmax_mode':True})
    res['RHC'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    print(pct_time_correct(res, 100))
    print(resource_report(res))
    
    
    fn = Trap()
    runner = SARunner(fn, dict(schedule = mr.GeomDecay(init_temp=15, decay=0.99),  
                                max_attempts = 20))
    res['SA'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    
    print(pct_time_correct(res, 100))
    print(resource_report(res))
    
    fn = Trap()
    runner = GARunner(fn, dict(max_attempts=10, pop_size=200, pop_breed_percent=0.75, mutation_prob=0.5,  elite_dreg_ratio=0.9))
    res['GA'] = suite.test(runner, runs)
    print(ranks(res))
    print(summary_scores(res))
    print(pct_time_correct(res, 100))
    print(resource_report(res))
    
    
    fn = Trap()
    runner = MIMICRunner(fn, dict(keep_pct=0.1, pop_size=200, max_attempts=5))
    res['MIMIC'] = suite.test(runner, runs)
    printdf(ranks(res), 'ranks-tr')
    printdf(summary_scores(res), "summ-tr")
    printdf(pct_time_correct(res, 94), "score-tr")
    printdf(resource_report(res), "resource-tr")
    
    with open('tr-{}.pkl'.format(N_), 'wb') as f:
        pkl.dump(res, f)
    


if __name__=='__main__':
    runs = 50
    gs = False
    
    if gs:
            
        suite = TestSuite(0)
        fn = Trap()
        runner = SARunner(fn, dict(schedule = mr.GeomDecay(init_temp=2, decay=0.8),  
                                    max_attempts = 20, equality=True))
        
        search_res = grid_search(suite, runner, param_grid=dict(schedule__init_temp=[0.001, 0.01, 0.1, 1, 10, 100, 500, 1000], 
                                                                schedule__decay=[0.9999, 0.9999, 0.999, 0.99, 0.9, 0.8, 0.5]), runs=runs)
        s = sorted(search_res.items(), key=lambda x: x[1], reverse=True)
        for line in s:
            print(line)
    
    ###
    for i in [10, 20, 30, 40, 50, 60]:
        runner(runs, i)
        
        
        
