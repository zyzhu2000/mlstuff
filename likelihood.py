import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


N = 64

seed = np.random.randint(65565)
print(seed)
np.random.seed(seed)

perm = np.arange(N)
np.random.shuffle(perm)

probs = np.random.uniform(0.4, 0.6, 2*N+3)

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
    v2 = fn_fitness2(state)
    assert abs(v2-v*1e18)<0.001
    return v*1e18

fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)
problem_cust.set_mimic_fast_mode(True)

np.random.seed(0)

if 0:
    
    best_state, best_fitness, curve = mr.random_hill_climb(problem_cust, restarts= 1, argmax_mode=True,   
                                                           random_state = 1, curve=True)
    
    print(best_state)
    print(best_fitness)
    
    plt.plot(curve)
    plt.show()
    
    
    best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = mr.GeomDecay(0.1, 0.9, 0.0001), 
                                                          max_attempts = 20, max_iters = 10000, 
                                                          random_state = 1, curve=True)
    
    print(best_state)
    print(best_fitness)
    
    plt.plot(curve)
    plt.show()
    
    
    best_state, best_fitness, curve = mr.genetic_alg(problem_cust, random_state = 2, max_attempts=100, pop_size=500, 
                                                     elite_dreg_ratio=0.9, curve=True)
    
    print(best_state)
    print(best_fitness)
    
    plt.plot(curve)
    plt.show()


best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=100,  curve=True, keep_pct=0.5, pop_size=1000)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
