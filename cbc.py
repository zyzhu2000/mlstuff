import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


N = 80
effect = np.ones(N)

M = 5


    
def fn_fitness(state:np.ndarray):
    v = 0
    s = 0
    for i in range(N//2):
        s = int(state[i])^int(state[-i-1])^s
        v += s
    return v
    

fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)

np.random.seed(0)
init_state = np.random.randint(0, 2, size=N)

print(fn_fitness(init_state))


best_state, best_fitness, curve = mr.random_hill_climb(problem_cust, restarts= 100,    init_state = init_state, random_state = 1, curve=True)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()



# Define decay schedule
schedule = mr.ExpDecay()
# Solve using simulated annealing - attempt 1

#init_state[:] = 0
#init_state[:5] = 1
print(fn_fitness(init_state))

best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 100, max_iters = 10000, 
                                                      init_state = init_state, random_state = 1, curve=True)

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

best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=100,  curve=True, keep_pct=0.2, pop_size=500)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
