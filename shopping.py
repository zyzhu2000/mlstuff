import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


N = 40
np.random.seed(888)
utility = np.random.uniform(low=0, high=1, size=N)
budget = N//3

l = list(reversed(sorted(utility)))
print(sum(l[:budget]))
           
    
def fn_fitness(state:np.ndarray):
    if np.sum(state)>budget:
        v = 0
    else:
        v = np.dot(state, utility)
    return v
    

fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)



#best_state, best_fitness, curve = mr.random_hill_climb(problem_cust, restarts= 5,    random_state = 1, curve=True)

#print(best_state)
#print(best_fitness)

#plt.plot(curve)
#plt.show()

np.random.seed(0)
init_state = np.random.randint(0, 2, size=N)

print(fn_fitness(init_state))

best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = mr.GeomDecay(init_temp=0.3, decay=0.9999, min_temp=0.001), 
                                                      max_attempts = 200, max_iters = 10000, 
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

best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=100,  curve=True, keep_pct=0.2, pop_size=200)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
