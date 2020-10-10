import numpy as np, pickle as pkl
import typing, itertools, importlib
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
    def set_params(self, params):
        self.params = params
    def update_params(self, params):
        self.params.update(params)
    
    
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
        
        rec = dict(state=best_state, fitness=best_fitness, curve=self.counter.curves, iterations=self.counter.total_iterations, 
                   evaluations=self.fn.count)
        return rec
    
class SARunner(Runner):
    def __init__(self, fn:FitnessFunction, params):
        self.fn = fn
        self.params = params
    
    def run(self, random_state):
        self.fn.reset()
        problem = self.fn.get_problem()
                
            
        def sa(problem, **kargs):
            d = {}
            d2 = {}
            for k in kargs:
                if '__' in k:
                    _, p = k.split('__')
                    d2[p] = kargs[k]
                else:
                    d[k] = kargs[k]
            schedule = mr.GeomDecay(**d2)
            if 'schedule' in d:
                del d['schedule']
                
            return mr.simulated_annealing(problem, schedule=schedule, **d)
        
        best_state, best_fitness, curve = sa(problem, random_state = random_state, curve=True, **self.params)
        
        
        rec = dict(state=best_state, fitness=best_fitness, curve=curve, iterations=len(curve), 
                   evaluations=self.fn.count)
        return rec

class GARunner(Runner):
    def __init__(self, fn:FitnessFunction, params):
        self.fn = fn
        self.params = params
    
    def run(self, random_state):
        self.fn.reset()
        
        problem = self.fn.get_problem()
        
        best_state, best_fitness, curve = mr.genetic_alg(problem, random_state = random_state, curve=True, **self.params)
        
        
        rec = dict(state=best_state, fitness=best_fitness, curve=curve, iterations=len(curve), 
                   evaluations=self.fn.count)
        return rec


class MIMICRunner(Runner):
    def __init__(self, fn:FitnessFunction, params):
        self.fn = fn
        self.params = params
    
    def run(self, random_state):
        self.fn.reset()
        
        problem = self.fn.get_problem()
        
        best_state, best_fitness, curve = mr.mimic(problem, random_state = random_state,  curve=True, **self.params)
        
        rec = dict(state=best_state, fitness=best_fitness, curve=curve, iterations=len(curve), 
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

def summary_scores(rec):
    algos = list(rec.keys())
    
    d = {}
    for i, a in enumerate(algos):
        x = rec[a]['fitness']
        d[a] = [np.mean(x), np.std(x), np.min(x), np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75), np.max(x)]
    df = pd.DataFrame.from_dict(d, orient='index', columns=['mean', 'std', 'min', '25 pct', '50 pct', '75 pct', 'max'])
    return df

def load_model(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)
    

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


def grid_search(suite:TestSuite, runner:Runner, param_grid:dict, runs:int, is_product=True):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    if is_product:
        gen_params = itertools.product(*values)
    else:
        l = [param_grid[k] for k in keys]
        gen_params = zip(*l)
        
    r = {}    
    for params in  gen_params:
        d = dict(zip(keys, params))
        runner.update_params(d)
        rec = suite.test(runner, runs)
        r[params] = np.mean(rec['fitness'])
    return r
    
def get_curves(raw_curves, runner_type):
    l = []
    M = 0
    for curve in raw_curves:
        if runner_type.lower()=='rhc':
                best_curve = None
                best_val = float('-inf')
                best_length = None
                for c in curve:
                    if c[-1] > best_val or c[-1] == best_val and best_length<len(c):
                        best_length = len(c)
                        best_val = c[-1]
                        best_curve = c
                        
                curve = best_curve

        l.append(curve)
        M = max(M, len(curve))
    
    p33 = []
    p66 = []
    p50 = []
    mean = []
    std = []
    for j in range(M):
        p = []
        for i in range(len(l)):
            if j<len(l[i]):
                p.append(l[i][j])
            else:
                p.append(l[i][-1])
        p33.append(np.percentile(p, 33))
        p66.append(np.percentile(p, 66))
        p50.append(np.percentile(p, 50))
        mean.append(np.mean(p))
        std.append(np.std(p))
    
    return {'p33':p33, 'p66': p66, 'p50': p50, 'mean': mean, 'std': std}
    
    
        
def make_curve(suite:TestSuite, runner:Runner, param_grid:dict, runs:int, is_product=True, cutoff=0.3, extend=False):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    curves = {}
    
    if is_product:
        gen_params = itertools.product(*values)
    else:
        l = [param_grid[k] for k in keys]
        gen_params = zip(*l)
        
        
    for params in  gen_params:
        d = dict(zip(keys, params))
        runner.update_params(d)
        rec = suite.test(runner, runs)
        l = []
        M = 0
        for curve in rec['curves']:
            if isinstance(runner, RHCRunner):
                best_curve = None
                best_val = float('-inf')
                best_length = None
                for c in curve:
                    if c[-1] > best_val or c[-1] == best_val and best_length<len(c):
                        best_length = len(c)
                        best_val = c[-1]
                        best_curve = c
                        
                curve = best_curve
            l.append(curve)
            M = max(M, len(curve))
        
        p33 = []
        p66 = []
        p50 = []
        mean = []
        std = []
        m = 0
        for j in range(M):
            p = []
            for i in range(len(l)):
                if j<len(l[i]):
                    p.append(l[i][j])
                elif extend:
                    p.append(l[i][-1])
                    
            m = max(m, len(p))
            if len(p)>cutoff*m:
                p33.append(np.percentile(p, 33))
                p66.append(np.percentile(p, 66))
                p50.append(np.percentile(p, 50))
                mean.append(np.mean(p))
                std.append(np.std(p))
        curves[params] = {'p33':p33, 'p66': p66, 'p50': p50, 'mean': mean, 'std': std, 
                          'time': np.mean(rec['time']), 'eval': np.mean(rec['evaluations'])}
    return curves

def run_grid(fn, param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    res = {}
    for params in  itertools.product(*values):
        print('Running {}'.format(params))
        d = dict(zip(keys, params))
        res[params] = fn(**d)
    return keys, res
    
        
def printdf(df:pd.DataFrame, filename=None):
    if filename is not None:
        df.to_csv(filename + ".csv")
        df.to_latex(filename + ".tex")
    print(df)
    
    
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
    
    
    
    
