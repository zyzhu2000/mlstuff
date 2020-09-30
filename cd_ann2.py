from sklearn.model_selection import KFold
import itertools, time, datetime
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
import sklearn.base
import imblearn as im
from sklearn import tree
import pickle as pkl

from  classifiers import *
from ann import *
from credit_default import CreditDefault_ANN
import mlrose_hiive as mr
from mlrose_hiive.algorithms.decay import GeomDecay

with open('ann.pkl', 'rb') as f:
    model = pkl.load(f)

    
X_train = model.trained_model['X_train']
y_train = model.trained_model['y_train']
X_test =  model.trained_model['X_test']
y_test =  model.trained_model['y_test']

np.random.seed(0)
def xavier_init(n1, n2):
    a  = np.sqrt(6/(n1+n2))
    return np.random.uniform(low=-a, high=a, size=n1*n2)

init_weights  = np.concatenate([xavier_init((X_train.shape[1]+1), 6), xavier_init(6, 1)])

if 0:
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'gradient_descent', 
                                     max_iters = 1000, bias = True, is_classifier = True, 
                                     learning_rate = 1e-5, early_stopping = True, 
                                     clip_max = 5, max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.show()
    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)


if 0:
    
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'random_hill_climb', 
                                     max_iters = 1500, bias = True, is_classifier = True, 
                                     learning_rate = 0.15, restarts=10, early_stopping = True, 
                                      max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.show()
    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)

if 1:
    
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'simulated_annealing', 
                                     schedule=GeomDecay(0.001,min_temp=0.00),
                                     max_iters = 1500, bias = True, is_classifier = True, 
                                     learning_rate = 0.15, restarts=10, early_stopping = True, 
                                      max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.show()
    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)

if 1:
    
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'genetic_alg', 
                                     max_iters = 200, bias = True, is_classifier = True, 
                                     learning_rate = 1e-6,  early_stopping = True, pop_size=400, clip_max=2,
                                      max_attempts = 10, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.show()
    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)


ii = 0

