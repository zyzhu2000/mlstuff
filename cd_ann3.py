from sklearn.model_selection import KFold
import itertools, time, datetime, functools
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from mlrose_hiive.algorithms.decay import GeomDecay
from loader import DataLoader
from test_harness import run_grid


loader = DataLoader()
data = loader.load("CreditDefault")

X = data.X
y = data.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8888)
X_valid = data.X_valid
y_valid = data.y_valid

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
X_valid = (X_valid - mean) / std

def xavier_init(n1, n2):
    a  = np.sqrt(6/(n1+n2))
    return np.random.uniform(low=-a, high=a, size=n1*n2)

def fn_init_weights():
    return np.concatenate([xavier_init((X_train.shape[1]+1), 6), xavier_init(6, 1)])


def train_gd(learning_rate=1e-5):
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'gradient_descent', 
                                     max_iters = 2000, bias = True, is_classifier = True, 
                                     learning_rate = learning_rate, early_stopping = True, 
                                     clip_max = 5, max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    y_valid_pred = nn_model2.predict(X_valid)
    y_valid_accuracy = f1_score(y_valid, y_valid_pred, average="macro")
    
    return nn_model2, y_train_accuracy, y_valid_accuracy
    

def train_rhc(learning_rate=0.15, restarts=10):
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'random_hill_climb', 
                                     max_iters = 2000, bias = True, is_classifier = True, 
                                     learning_rate = learning_rate, restarts=restarts, early_stopping = True, 
                                     clip_max= 1.5,
                                      max_attempts = 100, random_state = 3, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=fn_init_weights)
    
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    y_valid_pred = nn_model2.predict(X_valid)
    y_valid_accuracy = f1_score(y_valid, y_valid_pred, average="macro")
    
    return nn_model2, y_train_accuracy, y_valid_accuracy


def train_sa(learning_rate=0.3, T=1, decay=0.99, early_stopping = True):
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'simulated_annealing', 
                                     clip_max= 1.5,
                                     schedule=GeomDecay(T,min_temp=0.0001, decay=decay),
                                     max_iters = 2000, bias = True, is_classifier = True, 
                                     learning_rate = learning_rate, restarts=10, early_stopping = early_stopping, 
                                      max_attempts = 100, random_state = 8, curve=True)
    nn_model2.fit(X_train, y_train, init_weights=init_weights)
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    y_valid_pred = nn_model2.predict(X_valid)
    y_valid_accuracy = f1_score(y_valid, y_valid_pred, average="macro")
    
    return nn_model2, y_train_accuracy, y_valid_accuracy
    
def train_ga(learning_rate=0.2, pop_size=200, clip_max=1.5, mutation_prob=0.05, pop_breed_pct=0.75, early_stopping = True, max_iters=400,
             max_attempts=10) :
    nn_model2 = mr.NeuralNetwork(hidden_nodes = [6], activation = 'relu', 
                                     algorithm = 'genetic_alg', 
                                     max_iters = max_iters, bias = True, is_classifier = True, 
                                     learning_rate = learning_rate,  early_stopping = early_stopping, pop_size=pop_size, clip_max=clip_max,
                                     mutation_prob=mutation_prob,
                                      max_attempts = max_attempts, random_state = 3, curve=True, pop_breed_percent=pop_breed_pct)
    init = np.random.uniform(-1.5,1.5, size=init_weights.shape)
    nn_model2.fit(X_train, y_train, init_weights=init)
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train)
    
    y_train_accuracy = f1_score(y_train, y_train_pred, average="macro")
    y_valid_pred = nn_model2.predict(X_valid)
    y_valid_accuracy = f1_score(y_valid, y_valid_pred, average="macro")
    
    return nn_model2, y_train_accuracy, y_valid_accuracy
    
    
def save(filename, obj):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)

def show_res(res):
    keys, res2 = res
    for k in res2:
        print(k, res2[k][1], res2[k][2])

        
np.random.seed(0)

init_weights  = np.concatenate([xavier_init((X_train.shape[1]+1), 6), xavier_init(6, 1)])
        
if __name__=='__main__':
    
        
    
    if 0:
        res = run_grid(train_gd, param_grid=dict(learning_rate=[1e-6, 5e-6, 1e-5, 1.5e-5, 2e-5, 3e-5, 4e-5, 5e-5]))
        save("gd.pkl", res)
        show_res(res)
    
    if 0:
        res = run_grid(train_rhc, param_grid=dict(learning_rate=[0.4, 0.45, 0.5, 0.55,  0.6]))
        save("rhc.pkl", res)
        show_res(res)
    
    if 0:
        res = run_grid(train_sa, param_grid=dict(T=[0.5, 1, 1.5, 2], decay=[0.7, 0.75, 0.8, 0.9, 0.99, 0.999]))
        save("sa.pkl", res)
        show_res(res)
        
    if 0:
        res = run_grid(train_sa, param_grid=dict(learning_rate=[0.25, 0.3, 0.35]))
        save("sa.pkl", res)
        show_res(res)
    
    
    if 0:
        res = run_grid(train_ga, param_grid=dict(pop_size=[400, 600, 800]))
        save("ga-pop.pkl", res)
        show_res(res)
    
    if 0:
        res = run_grid(train_ga, param_grid=dict(learning_rate=[1e-6, 1e-5, 1e-4, 1e-3]))
        save("ga-lr.pkl", res)
        show_res(res)
    
    if 0:
        res = run_grid(train_ga, param_grid=dict(pop_breed_pct=[0.725, 0.75, 0.775]))
        save("ga-mate.pkl", res)
        show_res(res)
    
    if 1:
        res = run_grid(train_ga, param_grid=dict(mutation_prob=[0.05] ))
        save("ga-mute.pkl", res)
        show_res(res)
    
    if 0:
        res = run_grid(train_ga, param_grid=dict(clip_max=[0.5, 1, 2, 2.5, 3]))
        save("ga-max.pkl", res)
        show_res(res)
        
        
    if 0:
        res = run_grid(train_ga, param_grid=dict(learning_rate=[0.22]))
        save("ga-max.pkl", res)
        show_res(res)        

    
    ii = 0

