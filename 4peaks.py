import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt


N = 50
T = 2

def head(state, val):
    idx = np.nonzero(state==val^1)[0]
    if len(idx)==0:
        head1 = len(state)
    else:
        head1 = idx[0] 
    return head1

def tail(state, val):
    return head(state[::-1], val)

def fn_fitness(state:np.ndarray):
    head1 = head(state, 1)
    tail0 = tail(state, 0)
    
    reward = 0
    if tail0 > T and head1 >T:
        reward = N
    v = max(tail0, head1)
    return reward + v



fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = mr.DiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True, max_val = 2)


# Define decay schedule
schedule = mr.ExpDecay(exp_const=0.001)
# Solve using simulated annealing - attempt 1

init_state = np.zeros(N)
best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 100, max_iters = 1000, 
                                                      init_state = init_state, random_state = 1, curve=True)
print(fn_fitness(init_state))
print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()

best_state, best_fitness, curve = mr.genetic_alg(problem_cust, random_state = 2, curve=True, max_attempts=100)
print(fn_fitness(init_state))
print(best_state)
print(best_fitness)


best_state, best_fitness, curve = mr.mimic(problem_cust, random_state = 2, max_attempts=10,  curve=True, keep_pct=0.2)

print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()