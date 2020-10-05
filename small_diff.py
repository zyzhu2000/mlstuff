import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


N = 20

n0 = np.arange(N, dtype=float)
weights = np.random.uniform(-1e-3, 1e-3, size=N-2)
    
def fn_fitness(state:np.ndarray):
    v = np.dot(weights, state[1:-1])
    if (state[0] + state[-1])==0:
        v = -v
        
    return v + 1

fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)

np.random.seed(0)


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


best_state, best_fitness, curve = mr.genetic_alg(problem_cust, random_state = 65655, max_attempts=100, pop_size=500, 
                                                 elite_dreg_ratio=1.0, curve=True, mutation_prob=0)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()

best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=100,  curve=True, keep_pct=0.5, pop_size=1000)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
