import matplotlib.pyplot as plt

import numpy as np
import logging, time, itertools
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve, learning_curve 
from sklearn.model_selection import cross_val_predict, cross_val_score, check_cv
from sklearn.model_selection import train_test_split, StratifiedKFold, BaseCrossValidator
from sklearn.metrics import check_scoring
from sklearn.ensemble import AdaBoostClassifier
import sklearn.metrics
import sklearn.base
import PIL
from PIL import Image
from skimage.filters import threshold_otsu
from tabulate import tabulate
from loader import DataLoader

g_interactive = True
verbose = 1

class ClassifierBase(object):
    def __init__(self):
        self.trained_model = self.model = None
        self.title = 'base'
        self.data_name = None
        
    def make_model(self, hp):
        pass
    
    def get_init_params(self):
        pass
    def get_cv_params(self):
        pass
    
    def get_abbrev_name(self):
        return  'base'
    
    def get_long_name(self):
        return 'Base Model'
    
    def get_cv_class(self):
        return GridSearchCV
    
    def get_scorer_name(self):
        return 'accuracy'
    
    def preprocess(self, X_train, y_train, X_test, y_test):
        return X_train, y_train, X_test, y_test
    
    def get_data_params(self):
        return None
    def get_data(self):
        loader = DataLoader()
        data = loader.load(self.data_name, self.get_data_params())
        
        if not hasattr(data, 'X_train'):
            X = data.X
            y = data.y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8888)
        else:
            X_train = data.X_train
            y_train = data.y_train
            X_test = data.X_test
            y_test = data.y_test
        return X_train, y_train, X_test, y_test
        
    def fit_model(self, n_folds=4, X_train2=None, y_train2=None, params=None, param_grid=None):
        '''
        Overall function to run a model
        1. load data
        2. preprocessing
        3. cross-validation and training
        6. accuracy report
        '''
        logging.info('Fitting the model...')
        
        X_train, y_train, X_test, y_test = self.get_data()
        
        if X_train2 is not None:
            X_train = X_train2
            y_train = y_train2
            
        
        X_train, y_train, X_test, y_test = self.preprocess(X_train, y_train, X_test, y_test)
        
        if param_grid is None:
            param_grid = self.get_cv_params()
        
        if params is not None:
            self.model.set_params(**params)
            param_grid = {k:v for k, v in param_grid.items() if k not in params}
        
        
        cv_class = self.get_cv_class()
        
        validator = StratifiedKFold(n_splits=n_folds, shuffle=False)
        cv = cv_class(self.model, param_grid, n_jobs=-1, cv=validator, scoring=self.get_scorer_name(), verbose=verbose)
        cv.fit(X_train, y_train) # cross validation
        
        logging.info("Best parameter (CV score=%0.3f):" % cv.best_score_)
        logging.info("Best parameters: {}".format(cv.best_params_))
        logging.info("Refit time: {}".format(cv.refit_time_))
        
        
        d = {'X_train': X_train,
             'y_train': y_train,
             'X_test': X_test,
             'y_test': y_test,
             'n_folds': n_folds,
             'cv': cv,
             #'val_conf': cv_conf_mat,
             #'test_conf': conf_mat
        }
        self.trained_model = d
        return d
    
    def eval_performance(self, estimator=None, reestimate=False):
        logging.info('Evaluating performance...')
        
        X_train = self.trained_model['X_train']
        y_train = self.trained_model['y_train']
        X_test = self.trained_model['X_test']
        y_test = self.trained_model['y_test']        
        
        n_folds = self.trained_model['n_folds']
        
        
        if estimator is None:
            estimator = self.trained_model['cv'].best_estimator_
        elif reestimate:
            estimator.fit(X_train, y_train)
        
        
        validator = StratifiedKFold(n_splits=n_folds, shuffle=False)
        cv_score = cross_val_score(estimator, X_train, y_train, cv=validator, n_jobs=-1, scoring=self.get_scorer_name()).mean()

        scorer = sklearn.metrics.get_scorer(self.get_scorer_name())        
        test_score = scorer(estimator, X_test, y_test)        
        train_score = scorer(estimator, X_train, y_train)
        
        logging.info('Training score={}'.format(train_score))
        logging.info('CV score={}'.format(cv_score))
        logging.info('test score={}'.format(test_score))
        
    
    
    def analyze_misclassifications(self):
        '''
        This method makes it easy to analyze the misclassifications instance by instance
        It splits the big training set into training and validation, train the model on training using the same
        hyperparameters, and output the classificaition results.
        '''
        model = self.trained_model['cv']
        X_train = self.trained_model['X_train']
        y_train = self.trained_model['y_train']
        n_folds = self.trained_model['n_folds']
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=1/n_folds, random_state=8888)
        
        model2 = self.make_model(model.best_estimator_.get_params())
        model2.fit(X_train2, y_train2)
        
        pred = model2.predict(X_test2)
        
        d = { 'pred': pred, 'X_train':X_train2, 'y_train':y_train2, 'X_test':X_test2, 'y_test': y_test2}
        self.analysis = d
        return d
        
    
    def make_validation_curve(self, pname,  filename, title, xlabel, ylabel="Error", rng=None, transform=None, ax=None, suffix='',show=False,
                              xscale=None, shuffle=False, rounds=1, random_state=0, flip=False):
        logging.info('Creating validation curve...')
        
        model = self.trained_model['cv']
        X_train = self.trained_model['X_train']
        y_train = self.trained_model['y_train']
        n_folds = self.trained_model['n_folds']
        
        if rng is None:
            rng = self.get_cv_params()[pname]
        
        train_scores0 = np.zeros((rounds, len(rng), n_folds))
        valid_scores0 = np.zeros((rounds, len(rng), n_folds))
        
        np.random.seed(random_state)
        
        for r in range(rounds):
            if shuffle:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
            else:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
            if hasattr(self, 'validation_curve'):
                train_scores, valid_scores = self.validation_curve(model.best_estimator_, X_train, y_train, 
                                                          param_name=pname, param_range=rng, scoring=self.get_scorer_name(),
                                                          cv=cv, n_jobs=-1, verbose=verbose)    
            else:
                train_scores, valid_scores = validation_curve(model.best_estimator_, X_train, y_train, 
                                                          param_name=pname, param_range=rng, scoring=self.get_scorer_name(),
                                                          cv=cv, n_jobs=-1, verbose=verbose)    
            train_scores0[r, ...] = train_scores
            valid_scores0[r, ...] = valid_scores
            
        train_scores = train_scores0.mean(axis=0)
        valid_scores = valid_scores0.mean(axis=0)
        
        if flip:
            ts = 1-train_scores.mean(axis=1)
            vs = 1-valid_scores.mean(axis=1)
        else:
            ts = train_scores.mean(axis=1)
            vs = valid_scores.mean(axis=1)

        r2 = rng
        if transform is not None:
            fn = transform
            r2 = fn(rng)
        
        if ax is None:
            fig, ax = plt.subplots()
        plt.sca(ax)
        
            
        ax.plot(r2, ts, label='Training ' + suffix, alpha=0.7)
        ax.plot(r2, vs, label='Validation ' + suffix, alpha=0.7)
        ax.grid(True)
        ax.legend()
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        
        if xscale:
            ax.set_xscale(xscale)
        
        if filename:
            plt.savefig(filename)
                    
        return r2, ts, vs
    
    def make_learning_curve(self, filename, title, xlabel='Training Data', ylabel="Error", ax=None, suffix='', cv=5, rounds=1, 
                            shuffle=True, random_state=0, train_sizes=np.linspace(0.1, 1.0, 5), flip=False):
        logging.info('Creating learning curve...')
        model = self.trained_model['cv']
        X_train = self.trained_model['X_train']
        y_train = self.trained_model['y_train']
        
        np.random.seed(random_state)
        
        
        
        train_scores0 = np.zeros((rounds, len(train_sizes), cv))
        valid_scores0 = np.zeros((rounds, len(train_sizes), cv))
        
        
        for r in range(rounds):
            if hasattr(self, 'learning_curve'):
                train_sizes0, train_scores, valid_scores, fit_times, score_times =  self.learning_curve(
                    model.best_estimator_, X_train, y_train, shuffle=shuffle, 
                    return_times=True,  cv=cv, verbose=verbose, n_jobs=-1, scoring=self.get_scorer_name(), train_sizes=train_sizes)             
            else:
                train_sizes0, train_scores, valid_scores, fit_times, score_times =  learning_curve(
                    model.best_estimator_, X_train, y_train, shuffle=shuffle,
                    return_times=True,  cv=cv, verbose=verbose, n_jobs=-1, scoring=self.get_scorer_name(), train_sizes=train_sizes) 
            
            train_scores0[r, ...] = train_scores
            valid_scores0[r, ...] = valid_scores
        
        train_scores = train_scores0.mean(axis=0)
        valid_scores = valid_scores0.mean(axis=0)
        
        
        if ax is None:
            fig, ax = plt.subplots()
        plt.sca(ax)
        
        assert isinstance(ax, plt.Axes)
        
        if flip:
            ts = 1-train_scores.mean(axis=1)
            vs = 1-valid_scores.mean(axis=1)
        else:
            ts = train_scores.mean(axis=1)
            vs = valid_scores.mean(axis=1)
        
        if title:
            ax.set_title(title)
           
        ax.plot(train_sizes0, ts, label='Training ' + suffix,  alpha=0.7)
        ax.plot(train_sizes0, vs, label='Validation ' + suffix, alpha=0.7)
        ax.grid(True)
        ax.legend()
        
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        
        if filename:
            plt.sca(ax)
            plt.savefig(filename)
    
    def make_tc_curve(self, rounds):
        logging.info('Creating time complexity curve...')
        
        model = self.trained_model['cv']
        X = self.trained_model['X_train']
        y = self.trained_model['y_train']
        
        
        train_size = np.arange(0.1, 1.1, 0.1)
        test_size = 1 - train_size
        
        tr_sizes = np.zeros((rounds, len(train_size)))
        tr_time = np.zeros((rounds, len(train_size)))

        te_sizes = np.zeros((rounds, len(train_size)))
        te_time = np.zeros((rounds, len(train_size)))
        
        
        
        estimator = sklearn.base.clone(model.best_estimator_)
        for r in range(rounds):
            for i, ts in enumerate(test_size):
                estimator = sklearn.base.clone(estimator)
                if ts!=0:
                    X_train2, _, y_train2, _ = train_test_split(X, y, test_size=ts, random_state=8888)
                    X_train3, _, y_train3, _ = train_test_split(X, y, test_size=ts, random_state=8888)
                else:
                    X_train2 = X_test3 = X
                    y_train2 = y_train3 = y
            
                t = time.time()
                estimator.fit(X_train2, y_train2)
                t = time.time() - t
                tr_sizes[r, i] = len(y_train2)
                tr_time[r, i] = t
                #print(len(y_train2), estimator.n_iter_, estimator.n_iter_ * len(y_train2)/128 )
                
                
                t = time.time()
                estimator.predict(X_train3)
                t = time.time() - t
                te_sizes[r, i] = len(y_train3)
                te_time[r, i] = t
        
        tr_sizes0 = tr_sizes.mean(axis=0)
        te_sizes0 = te_sizes.mean(axis=0)
        tr_time0 = tr_time.mean(axis=0)
        te_time0 = te_time.mean(axis=0)
        return tr_sizes0, tr_time0, te_sizes0, te_time0

class FakeEstimator(object):
    def __init__(self, y_pred):
        self.y_pred = y_pred
    def predict(self, X):
        return self.y_pred

class BoostSearchCV(object):
    def __init__(self, estimator, param_grid, n_jobs, cv, scoring, verbose):
        self.param_grid = param_grid
        self.cv = cv
        self.verbose = verbose
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
    
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
        
    def fit(self, X, y):
        if isinstance(self.cv, BaseCrossValidator):
            kf = self.cv
            n_folds = kf.get_n_splits()
        else:
            kf = StratifiedKFold(n_splits=self.cv)
            n_folds = self.cv
        n_estimators = max(self.param_grid['n_estimators'])
        
        keys = [x for x in self.param_grid if x!= 'n_estimators']
        values = [self.param_grid[x] for x in self.param_grid if x!= 'n_estimators']
        
        fold_estimators = []
        for j in range(n_folds):
            estimators = []
            for params in  itertools.product(*values):
                d = dict(zip(keys, params))
                d['n_estimators'] = n_estimators
                estimator = sklearn.base.clone(self.estimator)
                estimator.set_params(**d)
                estimators.append(estimator)
                
            fold_estimators.append(estimators)
                
        
        def task(idx_train, idx_test, estimators, scoring):
            scr = np.zeros((len(fold_estimators[0]), n_estimators))
            
            X_train = X[idx_train,:]
            y_train = y[idx_train]
            X_test = X[idx_test,:]
            y_test = y[idx_test]
            
            scorer = sklearn.metrics.get_scorer(scoring)
            
            for i, est in enumerate(estimators):
                est.fit(X_train, y_train)
                for k, yp in enumerate(est.staged_predict(X_test)):
                    fake = FakeEstimator(yp)
                    scr[i, k] = scorer(fake, X_test, y_test)
                assert k == n_estimators-1
                
            return scr
        
        n_jobs = self.n_jobs
        
        res = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                    delayed(task)(idx_train, idx_test, fold_estimators[j], self.scoring) 
                    for k, (idx_train, idx_test) in enumerate(kf.split(X, y))
            )
            
        
        scores = np.zeros((n_folds, len(fold_estimators[0]), n_estimators))
        for j in range(n_folds):
            scores[j, ...] =res[j]
        
        self.scores = scores
        self.estimators = fold_estimators
        scores2 = scores.mean(axis=0)
        
        i_est = np.argmax(scores2.ravel())
        i, j = np.unravel_index(i_est, scores2.shape)
        est = fold_estimators[0][i]
        params = est.get_params().copy()
        params['n_estimators'] = j+1
        
        self.best_estimator_ = est
        self.best_estimator_.set_params(**params)
        self.best_score_ = scores2[i,j]
        self.best_params_ = {k:params[k] for k in self.param_grid}
        
        t = time.time()
        self.best_estimator_.fit(X, y)
        self.refit_time_ = time.time() - t
        
class BoostingVCMixin(object):
    def validation_curve(self, estimator, X, y, param_name, param_range, scoring, cv, n_jobs, verbose):
        if param_name!='n_estimators':
            return validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, 
                                    n_jobs=n_jobs, verbose=verbose, scoring=scoring) 
        
        estimator = sklearn.base.clone(estimator)
        n_estimators = max(param_range)
        estimator.set_params(n_estimators=n_estimators)
        
        if isinstance(cv, BaseCrossValidator):
            n_folds = cv.get_n_splits()
            kf = cv
        else:
            n_folds = cv
            kf = StratifiedKFold(n_splits=cv)
        
        
        scr_test = np.zeros((n_estimators, n_folds))
        scr_train = np.zeros((n_estimators, n_folds))
        
        def task(estimator, idx_train, idx_test, scoring):
            s_test = np.zeros((n_estimators))
            s_train = np.zeros((n_estimators))
            
            
            X_train = X[idx_train,:]
            y_train = y[idx_train]
            X_test = X[idx_test,:]
            y_test = y[idx_test]
            
            estimator.fit(X_train, y_train)
            
            scorer = sklearn.metrics.get_scorer(scoring)
            for k, yp in enumerate(estimator.staged_predict(X_test)):
                s_test[k] = scorer(FakeEstimator(y_test), X_test,  yp)
            assert k == n_estimators-1
            
            
            for k, yp in enumerate(estimator.staged_predict(X_train)):
                s_train[k] = scorer(FakeEstimator(y_train), X_train, yp)
            assert k == n_estimators-1
            
            
            return s_train, s_test
        
        res = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                    delayed(task)(sklearn.base.clone(estimator), idx_train, idx_test, scoring) 
                    for idx_train, idx_test in kf.split(X, y)
            )
    
        for k, (s_train, s_test) in enumerate(res):
            scr_train[:, k] = s_train
            scr_test[:, k] = s_test
        
        return scr_train, scr_test

class SVMLCMixin(object):
    def learning_curve(self, estimator, X, y, *, groups=None, cv=4,
                   train_sizes=np.linspace(0.1, 1.0, 5),  scoring=None, 
                   n_jobs=None, pre_dispatch="all", verbose=0, shuffle=False,
                   random_state=None, error_score=np.nan, return_times=False):
        
        if isinstance(cv, BaseCrossValidator):
            n_folds = cv.get_n_splits()
            kf = cv
        else:
            n_folds = cv
            kf = StratifiedKFold(n_splits=cv)

        
        
        
        if isinstance(estimator, SVC):
            C = estimator.get_params()['C']
            m = estimator.shape_fit_[0]            
        else:
            C = estimator.get_params()['model__C']
            m = estimator['model'].shape_fit_[0]

        def task(estimator, X_train, y_train, X_test, y_test, train_size, scoring):
            ts = int(train_size*X_train.shape[0])
            
            X_train2 = X_train[:ts, :]
            y_train2 = y_train[:ts]
            
            
            if isinstance(estimator, SVC):
                estimator.set_params(C= C*m/ts)
            else:
                estimator.set_params(model__C= C*m/ts)
                
            
            estimator.fit(X_train2, y_train2)
            scorer = sklearn.metrics.get_scorer(scoring)
            test_score = scorer(estimator, X_test, y_test)
            train_score = scorer(estimator, X_train2, y_train2)
            return ts, train_score, test_score
        
        
        n_ts = len(train_sizes)
        training_size = np.zeros((n_ts))
        training_score = np.zeros((n_ts, n_folds))
        test_score = np.zeros((n_ts, n_folds))
        
        #n_jobs = 1
        
        for j, (idx_train, idx_test) in enumerate(kf.split(X, y)):
            X_train = X[idx_train,:] ;  y_train = y[idx_train]
            X_test = X[idx_test,:] ;    y_test = y[idx_test]
            
            
            res = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                        delayed(task)(sklearn.base.clone(estimator), X_train, y_train, X_test, y_test, ts, scoring) 
                        for ts in train_sizes
                )
            for k, (ts, s_train, s_test) in enumerate(res):
                training_size[k] = ts
                training_score[k, j] = s_train
                test_score[k, j] = s_test

        return training_size, training_score, test_score, None, None
        

            
def make_tc_curve(model, title, filename):
    tr_sizes, tr_time, te_sizes, te_time = model.make_tc_curve(3)
    
    plt.figure()
    plt.plot(tr_sizes, tr_time, label='Training Time')
    plt.plot(te_sizes, te_time, label='Evaluation Time')
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Time (sec)')
    plt.savefig(filename)
    plt.close()
    logging.info('Time Complexity ' + title)
    logging.info('Training size: {}'.format(tr_sizes))
    logging.info('Training time: {}'.format(tr_time))
    logging.info('Evaluation size: {}'.format(tr_sizes))
    logging.info('Evaluation time: {}'.format(tr_time))    
            
           
class USPSBase(ClassifierBase):
    def eval_performance(self, estimator=None, reestimate=False):
        logging.info('Evaluating performance...')
        
        X_train = self.trained_model['X_train']
        y_train = self.trained_model['y_train']
        X_test = self.trained_model['X_test']
        y_test = self.trained_model['y_test']        
        
        n_folds = self.trained_model['n_folds']
        
        
        if estimator is None:
            estimator = self.trained_model['cv'].best_estimator_
        elif reestimate:
            estimator.fit(X_train, y_train)
        
        
        y_pred = cross_val_predict(estimator, X_train, y_train, cv=n_folds, n_jobs=-1)
        cv_conf_mat = sklearn.metrics.confusion_matrix(y_train, y_pred)
        cv_acc_all = np.diag(cv_conf_mat) /cv_conf_mat.sum(axis=1)
        cv_score = sklearn.metrics.accuracy_score(y_train, y_pred)
        
        
        pred = estimator.predict(X_test)
        
        score = sklearn.metrics.accuracy_score(y_test, pred)
        conf_mat = sklearn.metrics.confusion_matrix(y_test, pred)
        acc_all = np.diag(conf_mat) /conf_mat.sum(axis=1)
        
        y_pred0 = estimator.predict(X_train)
        conf_mat0 = sklearn.metrics.confusion_matrix(y_train, y_pred0)
        acc_all0 = np.diag(conf_mat0) /conf_mat0.sum(axis=1)
        score0 = sklearn.metrics.accuracy_score(y_train, y_pred0)
        
        
        headers = list(' 0123456789') + ['Total']
        row0 = ['Training'] + ["{:.2f}%".format(x) for x in acc_all0*100] + ["{:.2f}%".format(score0*100)]
        row1 = ['Validation'] + ["{:.2f}%".format(x) for x in cv_acc_all*100] + ["{:.2f}%".format(cv_score*100)]
        row2 = ['Test'] + ["{:.2f}%".format(x) for x in acc_all*100] + ["{:.2f}%".format(score*100)]
        
        
        logging.info(tabulate([row0, row1, row2], headers=headers, numalign="right"))
        
        row0 = ['Training'] + ["{:.2f}%".format(x) for x in acc_all0[0:6]*100] 
        row1 = ['Validation'] + ["{:.2f}%".format(x) for x in cv_acc_all[0:6]*100] 
        row2 = ['Test'] + ["{:.2f}%".format(x) for x in acc_all[0:6]*100] 
        
        headers = list(' 012345')
        fname = self.get_abbrev_name() + "-rpt0.tex"
        with open(fname, "w") as f2:
            print(tabulate([row0, row1, row2], headers=headers, tablefmt='latex', stralign="right",), file=f2)
        
        row0 = ['Training'] + ["{:.2f}%".format(x) for x in acc_all0[6:]*100] + ["{:.2f}%".format(score0*100)]
        row1 = ['Validation'] + ["{:.2f}%".format(x) for x in cv_acc_all[6:]*100] + ["{:.2f}%".format(cv_score*100)]
        row2 = ['Test'] + ["{:.2f}%".format(x) for x in acc_all[6:]*100] + ["{:.2f}%".format(score*100)]
        
        headers = list(' 6789') + ['Total']
        fname = self.get_abbrev_name() + "-rpt1.tex"
        with open(fname, "w") as f2:
            print(tabulate([row0, row1, row2], headers=headers, tablefmt='latex', stralign="right",), file=f2)
    
    
        

class USPS_kNN(USPSBase):
    def __init__(self, normalize=False):
        self.normalize = normalize
        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "USPS"
    
    def make_model(self, hp):
        return KNeighborsClassifier(**hp)
    
   
        
        
    def get_init_params(self):
        return {
            'n_neighbors': 5,
            'p': 2
        }
    
    def get_cv_params(self):
        return {
            'n_neighbors': np.arange(1,10),
            'p': [1, 1.25, 1.5, 2.0, 2.5],
            'weights':['uniform','distance']
        }
    def get_abbrev_name(self):
        return 'knn'
    
    def get_long_name(self):
        return 'kNN'

        
class USPS_SVCBase(USPSBase):
    def __init__(self):
        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "USPS"
    
    
    
    def make_model(self, hp):
        return SVC(**hp)
    
    
    def show_sv(self):
        model = self.trained_model['cv'].best_estimator_
        y_train = self.trained_model['y_train']
        logging.info('Total training data={} sv={}'.format(len(y_train), model.n_support_.sum()))
    
    def get_sv(self):
        model = self.trained_model['cv'].best_estimator_
        sv = model.support_vectors_
        return sv
    
    def test_sv(self):
        model = self.trained_model['cv'].best_estimator_
        idx = model.support_ 
        X_train = self.trained_model['X_train']
        y_train = self.trained_model['y_train']
        
        X_train2 = X_train[idx,:]
        y_train2 = y_train[idx]
        
        model2 = USPS_ANN()
        model2.fit_model(X_train2=X_train2, y_train2=y_train2)
    

        
    
class USPS_SVCLinear(USPS_SVCBase):
    
    def get_init_params(self):
        return    { 'kernel': 'linear', 'C': 1.0}

    
    def get_cv_params(self):
        return {
            #'C': np.logspace(np.log10(1/2000), np.log10(0.05), 20)
            'C': np.linspace(0.0005, 0.01, 20),
        }
    def get_abbrev_name(self):
        return 'svm-linear'
    
    def get_long_name(self):
        return 'SVM(Linear)'
    

class USPS_SVCRBF(USPS_SVCBase):
    
    def get_init_params(self):
        return    {
            'kernel': 'rbf',
            'C': 1.0,
        }
   
    def get_cv_params(self):
        return {
            #'C': np.logspace(np.log10(1/2000), np.log10(0.05), 20)
            'C': np.linspace(0.05, 5.0, 10),
            'gamma': np.logspace(np.log10(1e-3), np.log10(0.03), 10)
        }

        
    def get_abbrev_name(self):
        return 'svm-rbf'
    
    def get_long_name(self):
        return 'SVM(RBF)'
    


def setup_logging(log_file_name):
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

