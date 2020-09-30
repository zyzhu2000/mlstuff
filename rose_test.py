import mlrose_hiive
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
# Define alternative N-Queens fitness function for maximization problem



def queens_max(state):

    # Initialize counter
    fitness = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
               and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness += 1

    return fitness

# Check function is working correctly
state = np.array([1, 4, 1, 3, 5, 5, 2, 7])

# The fitness of this state should be 22
queens_max(state)

# Initialize custom fitness function object
fitness_cust = mlrose_hiive.CustomFitness(queens_max)

# Define optimization problem object
problem_cust = mlrose_hiive.DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize = True, max_val = 8)


# Define decay schedule
schedule = mlrose_hiive.ExpDecay()
# Solve using simulated annealing - attempt 1

init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 10, max_iters = 1000, 
                                                      init_state = init_state, random_state = 1)

print(best_state)