import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class TrapDiscreteOpt(mr.DiscreteOpt):
    def __init__(self, length, fitness_fn, maximize):
        super().__init__(length = length, fitness_fn=fitness_fn, maximize=maximize)
        
    def random_neighbor(self):
        # equalize the probability of 0->1 and 1->0
        neighbor = np.copy(self.state)
        if np.random.uniform()<0.05:
            for j in range(self.length):
                neighbor[j] ^= 1      
            return neighbor
        
        idx1 = np.nonzero(self.state==1)[0]
        if len(idx1)==0:
            idx = np.nonzero(self.state==0)[0]
        elif len(idx1)==self.length:
            idx = idx1
        elif np.random.uniform()>0.5:
            idx = np.nonzero(self.state==0)[0]
        else:
            idx = idx1
            
            
        neighbor = np.copy(self.state)    
        i = np.random.randint(0, len(idx))    
        neighbor[idx[i]] ^= 1
        return neighbor



N = 30

n0 = np.arange(N, dtype=float)
weights = 2**n0
    
def fn_fitness(state:np.ndarray):
    u = np.sum(state)
    a = 80
    b = 100
    z = 15+2
    
    if u<=z:
        v = a/z*(z-u)
    else:
        v = b/(N-z)*(u-z)
        
    return v 

fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = TrapDiscreteOpt(length = N, fitness_fn = fitness_cust, maximize=True)

np.random.seed(0)


best_state, best_fitness, curve = mr.random_hill_climb(problem_cust, restarts= 20, argmax_mode=True,   
                                                       random_state = 1, curve=True)

print(best_state)
print(best_fitness)

#plt.plot(curve)
#plt.show()


best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = mr.GeomDecay(0.01, 0.1, 0.0001), 
                                                      max_attempts = 10, max_iters = 10000, 
                                                      random_state = 209652397, curve=True, equality=True)

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
