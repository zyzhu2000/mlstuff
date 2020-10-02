import numpy as np
import time 
import scipy.stats
import pandas as pd
import mlrose_hiive as mr
from mrcounter import RHCCounter

class FitnessFunction(object):
    def __init__(self):
        self.count = 0
        
    def __call__(self, state):
        self.count += 1
    
    def reset(self):
        self.count = 0
    
    def get_problem(self):
        raise NotImplemented()
        
class Runner(object):
    def run(self, random_state):
        pass
    
class RHCRunner(Runner):
    def __init__(self, fn:FitnessFunction, params):
        self.fn = fn
        self.params = params
    
    def run(self, random_state):
        self.fn.reset()
        self.counter = RHCCounter()
        rhc_user_info = []
        problem = self.fn.get_problem()
        
        best_state, best_fitness, curve = mr.random_hill_climb(problem, random_state=random_state, curve=True, 
                                                               state_fitness_callback=self.counter, 
                                                               callback_user_info=rhc_user_info, **self.params)        
        
        rec = dict(state=best_state, fitness=best_fitness, curve=curve, iterations=self.counter.total_iterations, 
                   evaluations=self.fn.count)
        return rec
    
class SARunner(Runner):
    def __init__(self, fn:FitnessFunction, params):
        self.fn = fn
        self.params = params
    
    def run(self, random_state):
        self.fn.reset()
        self.counter = RHCCounter()
        problem = self.fn.get_problem()
        
        best_state, best_fitness, curve = mr.simulated_annealing(problem, random_state = random_state, curve=True, **self.params)
        
        
        rec = dict(state=best_state, fitness=best_fitness, curve=curve, iterations=self.counter.total_iterations, 
                   evaluations=self.fn.count)
        return rec

class GARunner(Runner):
    def __init__(self, fn:FitnessFunction, params):
        self.fn = fn
        self.params = params
    
    def run(self, random_state):
        self.fn.reset()
        self.counter = RHCCounter()
        problem = self.fn.get_problem()
        
        best_state, best_fitness, curve = mr.genetic_alg(problem, random_state = random_state, curve=True, **self.params)
        
        
        rec = dict(state=best_state, fitness=best_fitness, curve=curve, iterations=self.counter.total_iterations, 
                   evaluations=self.fn.count)
        return rec


class MIMICRunner(Runner):
    def __init__(self, fn:FitnessFunction, params):
        self.fn = fn
        self.params = params
    
    def run(self, random_state):
        self.fn.reset()
        self.counter = RHCCounter()
        problem = self.fn.get_problem()
        
        best_state, best_fitness, curve = mr.mimic(problem, random_state = random_state,  curve=True, **self.params)
        
        rec = dict(state=best_state, fitness=best_fitness, curve=curve, iterations=self.counter.total_iterations, 
                   evaluations=self.fn.count)
        return rec
    

class TestSuite(object):
    def __init__(self, random_state):
        self.random_state = random_state
    
    def test(self, model:Runner, runs:int):
        np.random.seed(self.random_state)
        fitness = []
        state = []
        iterations = []
        evaluations = []
        times = []
        curves = []
        
        seeds = np.random.randint(low=1, high=2**30, size=runs)
        for i, seed in enumerate(seeds):
            print('Test iteration {}/{}'.format(i+1, runs))
            t = time.time()
            d = model.run(seed)
            t2 = time.time()
            fitness.append(d['fitness'])
            state.append(d['state'])
            iterations.append(d['iterations'])
            evaluations.append(d['evaluations'])
            curves.append(d['curve'])
            times.append(t2-t)
        
        rec = dict(fitness=fitness, state=state, iterations=iterations, 
                   evaluations=evaluations, time=times, curves=curves)
        return rec

def ranks(rec):
    algos = list(rec.keys())
    
    ft = np.array([rec[a]['fitness'] for a in algos])
    
    R = np.zeros((len(algos), len(algos)), dtype=int)
    
    for i in range(ft.shape[1]): # for each experiment
        f = ft[:, i]
        ranks = scipy.stats.rankdata(-f, method='min') - 1
        
        for j in range(len(algos)):
            R[j, ranks[j]] += 1
    df = pd.DataFrame(R, index=algos, columns=np.arange(1, len(algos)+1))
    return df

def pct_time_correct(rec, truth):
    algos = list(rec.keys())
    
    l = []
    for i, a in enumerate(algos):
        pct = sum(1 for x in rec[a]['fitness'] if x>=truth)/len(rec[a]['fitness'])
        l.append(pct)
    df = pd.DataFrame(l, index=algos, columns=['% Max'])
    return df
        
def resource_report(rec):
    algos = list(rec.keys())
    d = {}
    for a in algos:
        t = rec[a]['time']
        e = rec[a]['evaluations']
        it = rec[a]['iterations']
        d[a] = [np.mean(t), np.std(t), np.mean(e), np.std(e), np.mean(it), np.std(it)]
    df = pd.DataFrame.from_dict(d, orient='index', columns=['Mean Time', 'Std Time', 'Mean Evals', 'Std Evals', 'Mean Iters', 'Std Iters'])
    return df
        
    
    
if __name__=='__main__':
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
        
    
    suite = TestSuite(0)
    
    fn = LessTraveled()
    runner = RHCRunner(fn, {'restarts':5})
    res = suite.test(runner, 3)
    print(res)
    
    r2 = {}
    r2['a1'] = res
    r2['a2'] = res
    
    print(ranks(r2))
    print(pct_time_correct(r2, 13))
    print(resource_report(r2))
    
    
    
    
