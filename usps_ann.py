import sys
from sklearn.model_selection import KFold
import itertools, time, datetime
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
import sklearn.base
from sklearn import tree
import pickle as pkl

from  classifiers import *
from ann import *
from credit_default import CreditDefault_ANN
import mlrose_hiive as mr

interactive = 0

mode = sys.argv[1]

with open('usps_ann.pkl', 'rb') as fh:
    model = pkl.load(fh)    


    
X_train = model.trained_model['X_train']
y_train = model.trained_model['y_train']
X_test =  model.trained_model['X_test']
y_test =  model.trained_model['y_test']

# One hot encode target values
one_hot = OneHotEncoder()

y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()

np.random.seed(0)
def xavier_init(n1, n2):
    a  = np.sqrt(6/(n1+n2))
    return np.random.uniform(low=-a, high=a, size=n1*n2)

init_weights  = np.concatenate([xavier_init((X_train.shape[1]+1), 120), xavier_init(120, 10)])

if mode=='gc':
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [120], activation = 'relu', 
                                     algorithm = 'gradient_descent', 
                                     max_iters = 1000, bias = True, is_classifier = True, 
                                     learning_rate = 1e-5, early_stopping = True, 
                                     clip_max = 5, max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.savefig('gc.png')
    if interactive:
        plt.show()
    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)


if mode=='rhc':
    
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [120], activation = 'relu', 
                                     algorithm = 'random_hill_climb', 
                                     max_iters = 4000, bias = True, is_classifier = True, 
                                     learning_rate = 0.15, restarts=10, early_stopping = True, clip_max=3, 
                                      max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.savefig('rhc.png')
    if interactive:
        plt.show()

    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)
    

if mode=='sa':
    
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [120], activation = 'relu', 
                                     algorithm = 'simulated_annealing', 
                                     max_iters = 10000, bias = True, is_classifier = True, clip_max=3,
                                     learning_rate = 0.15, restarts=10, early_stopping = True, 
                                      max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.savefig('sa.png')
    if interactive:
        plt.show()

    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)

if mode=='ga':
    
    # Initialize neural network object and fit object - attempt 2
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [120], activation = 'relu', 
                                     algorithm = 'genetic_alg', 
                                     max_iters = 200, bias = True, is_classifier = True, 
                                     learning_rate = 1e-6,  early_stopping = True, pop_size=200, clip_max=3,
                                      max_attempts = 20, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    plt.plot(nn_model2.fitness_curve)
    plt.savefig('ga.png')
    if interactive:
        plt.show()

    
    y_test_pred = nn_model2.predict(X_test)
    
    y_test_accuracy = f1_score(y_test, y_test_pred, average="macro")
    
    print(y_train_accuracy, y_test_accuracy)


ii = 0

