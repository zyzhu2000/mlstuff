import mlrose_hiive as mr
import numpy as np
import matplotlib.pyplot as plt

class RoundTable(mr.DiscreteOpt):
    """Class for defining travelling salesperson optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector. Must equal number of nodes in the
        tour.

    fitness_fn: fitness function object, default: None
        Object to implement fitness function for optimization. If :code:`None`,
        then :code:`TravellingSales(coords=coords, distances=distances)` is
        used by default.

    maximize: bool, default: False
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    coords: list of pairs, default: None
        Ordered list of the (x, y) co-ordinates of all nodes. This assumes
        that travel between all pairs of nodes is possible. If this is not the
        case, then use distances instead. This argument is ignored if
        fitness_fn is not :code:`None`.

    distances: list of triples, default: None
        List giving the distances, d, between all pairs of nodes, u and v, for
        which travel is possible, with each list item in the form (u, v, d).
        Order of the nodes does not matter, so (u, v, d) and (v, u, d) are
        considered to be the same. If a pair is missing from the list, it is
        assumed that travel between the two nodes is not possible. This
        argument is ignored if fitness_fn or coords is not :code:`None`.
    """

    def __init__(self, length, fitness_fn, maximize, max_val):
        mr.DiscreteOpt.__init__(self, length, fitness_fn, maximize, max_val=length)


    def random_neighbor(self):
        """Return random neighbor of current state vector.

        Returns
        -------
        neighbor: array
            State vector of random neighbor.
        """
        neighbor = np.copy(self.state)
        node1, node2 = np.random.choice(np.arange(self.length),
                                        size=2, replace=False)

        neighbor[node1] = self.state[node2]
        neighbor[node2] = self.state[node1]

        return neighbor



N = 10
chemistry = np.ones((N+1, N+1))
chemistry[2, 5 ] = chemistry[5, 2] = 10
chemistry[3, 5 ] = chemistry[5, 3] = -6
chemistry[4, 7 ] = chemistry[7, 4] = 3

chemistry = np.random.uniform(size=(N+1, N+1))
chemistry = (chemistry + chemistry.T)

def fn_fitness(state):
    i = N
    fitness = 0
    
    for s in state:
        fitness +=  chemistry[i, s]
        i = s
    fitness +=  chemistry[s, N]

    return fitness





fitness_cust = mr.CustomFitness(fn_fitness)

# Define optimization problem object
problem_cust = RoundTable(length = N, fitness_fn = fitness_cust, maximize=True, max_val = N)


# Define decay schedule
schedule = mr.ExpDecay()
# Solve using simulated annealing - attempt 1

init_state = np.arange(0, N)
best_state, best_fitness, curve = mr.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 10, max_iters = 1000, 
                                                      init_state = init_state, random_state = 1, curve=True)
print(fn_fitness(init_state))
print(best_state)
print(best_fitness)

plt.plot(curve)
plt.show()
