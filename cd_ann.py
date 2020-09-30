from sklearn.model_selection import KFold
import itertools, time, datetime
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import sklearn.base
import imblearn as im
from sklearn import tree
import pickle as pkl

from  classifiers import *
from ann import *
from credit_default import CreditDefault_ANN

now = str(datetime.datetime.now()).replace(':','')
setup_logging('log/' + now + '.log')


model = CreditDefault_ANN()
model.fit_model(5)
model.eval_performance()

with open('ann.pkl', 'wb') as f:
    pkl.dump(model, f)