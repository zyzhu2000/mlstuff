import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from mrcounter import *


N = 16
m1 = 2

masks = 2**np.arange(0, N)
    
def fn_fitness(state:np.ndarray):
    state = state.astype(int)
    x1 = state[0:m1]
    x2 = state[m1:]
    
    
    if all(x1==0):
        v = np.sum(1-x2) + 1
    else:
        v = np.sum(x2) 
    
    return float(v)

alpha = 3
def fn_fitness2(state:np.ndarray):
    state = state.astype(int)
    x1 = state[0:m1]
    x2 = state[m1:]
    n2 = len(x2)
    
    if all(x1==0):
        y = np.sum(1-x2)/n2
        v = y**(1/alpha) + 0.05
    else:
        y = np.sum(x2)/n2
        v = y**(alpha)
    
    return v
    

fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)


rhc_counter = RHCCounter()
rhc_user_info = []
best_state, best_fitness, curve = mr.random_hill_climb(problem_cust, restarts= 5,    random_state = 989978, curve=True, argmax_mode=True,
                                                       state_fitness_callback=rhc_counter, callback_user_info=rhc_user_info, max_attempts=2)
print(rhc_counter)
print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()

np.random.seed(0)
init_state = np.random.randint(0, 2, size=N)

print(fn_fitness(init_state))

best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = mr.GeomDecay(init_temp=20, decay=0.9995, min_temp=0.001), 
                                                      max_attempts = 200, max_iters = 10000, 
                                                       random_state = 68998, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()


best_state, best_fitness, curve = mr.genetic_alg(problem_cust, random_state = 2, max_attempts=100, pop_size=200, mutation_prob=0.2,
                                                 elite_dreg_ratio=0.95, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()

best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=100,  curve=True, keep_pct=0.2, pop_size=200)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
