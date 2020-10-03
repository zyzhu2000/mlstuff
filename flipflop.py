import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

N = 40

fitness_cust = mr.FlipFlop()



# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)

np.random.seed(0)
init_state = np.random.randint(0, 2, size=N)


best_state, best_fitness, curve = mr.random_hill_climb(problem_cust, restarts= 100,  max_attempts=100,  
                                                       init_state = init_state, random_state = 1, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()



# Define decay schedule
schedule = mr.ExpDecay(10, 0.00001)
# Solve using simulated annealing - attempt 1

#init_state[:] = 0
#init_state[:5] = 1


best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 100, max_iters = 10000, 
                                                      init_state = init_state, random_state = 1, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()


best_state, best_fitness, curve = mr.genetic_alg(problem_cust, random_state = 2, max_attempts=20, pop_size=500, 
                                                 elite_dreg_ratio=0.9, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()

best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=20,  curve=True, keep_pct=0.2, pop_size=500)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
