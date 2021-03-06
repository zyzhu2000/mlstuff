import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


N = 21
penalty = 1

def count_score(state, val):
    score = 0.0
    p = 0
    for i in range(len(state)):
        if state[i] == val:
            score += 1
            if i==0 or state[i]!=state[i-1]:
                p += penalty
    return score - p

def threholding(s):
    if s<4:
        return 0
    elif 4<=s<14:
        return 50
    else:
        return 100
        

def fn_fitness(state:np.ndarray):
    s0 = count_score(state, 0)
    s1 = count_score(state, 1)
    
    v0 = threholding(s0)
    v1 = threholding(s1)
    
    return v0 + v1
            

def fn_fitness2(state:np.ndarray):
    s0 = count_score(state, 0)
    s1 = count_score(state, 1)
    
    pv = np.array([s0-15, s1-15, s0-3, s1-3])
    p = scipy.stats.norm.cdf(pv/1)
    
    
    v = (p[0] + p[1])*(1000-100) + (p[2] + p[3])*100
    
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

best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = mr.GeomDecay(10, 0.999), 
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
