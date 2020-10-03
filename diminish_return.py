import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


N = 20
M = 3
value = np.array([1, 1.0001, 1.0002])
pos = np.arange(N)        
pos = np.mod(pos, M).astype(int)

def fn_fitness(state:np.ndarray):
    x = np.sum(state*value[pos])
    y = np.sum((1-state)*value[pos])
    
    v = 1.0/ (np.abs(x-10) + np.abs(y-10)+1)
    
    return v
            


fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)

np.random.seed(0)
init_state = np.random.randint(0, 2, size=N)


best_state, best_fitness, curve = mr.random_hill_climb(problem_cust, restarts= 100,  max_attempts=100,  
                                                       random_state = 1, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()



print(fn_fitness(init_state))

best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = mr.GeomDecay(1, 0.99), 
                                                      max_attempts = 100, max_iters = 10000, 
                                                      init_state = init_state, random_state = 1, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()


best_state, best_fitness, curve = mr.genetic_alg(problem_cust, random_state = 2, max_attempts=100, pop_size=200, 
                                                 elite_dreg_ratio=0.9, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()

best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=10,  curve=True, keep_pct=0.2, pop_size=500)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
