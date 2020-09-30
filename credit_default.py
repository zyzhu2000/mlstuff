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

class CreditDefaultBase(ClassifierBase):
    def get_scorer_name(self):
        return 'f1_macro'
        #return 'accuracy'

class CreditDefault_kNN(CreditDefaultBase):
    def __init__(self, reduce=False):
        self.reduce = reduce
        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "CreditDefault"
    
    def make_model(self, hp):
        #sampler = im.over_sampling.SMOTE(random_state=0)
        steps = [ #('feature_selection', SelectFromModel(RandomForestClassifier(), max_features=5)), 
                  #('smp', sampler),
                  ('scale', StandardScaler()),  
                  ('model', KNeighborsClassifier(**hp))
                  ]
        if self.reduce:
            steps = [('feature_selection', SelectFromModel(DecisionTreeClassifier(ccp_alpha=0.001225)))] + steps
        p = im.pipeline.Pipeline(steps=steps)        
        
        return p
        
        
    def get_init_params(self):
        return {
            'n_neighbors': 5,
            'p': 2
        }
    
    def get_cv_params(self):
        if not self.reduce:
            return {
                'model__n_neighbors': np.arange(1, 20, 2),
                'model__p': [1],
                'model__weights':['uniform']
            }
        return {
            'model__n_neighbors': np.arange(1, 40, 2),
            'model__p': [1],
            'model__weights':['uniform']
        }
        
    def get_abbrev_name(self):
        return 'knn-CreditDefault'
    
    def get_long_name(self):
        return 'kNN(CreditDefault)'

        
class CreditDefault_SVCBase(CreditDefaultBase, SVMLCMixin):
    def __init__(self):
        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "CreditDefault"
    
    def make_model(self, hp):
        steps = [ ('scale', StandardScaler()),  
                  ('model', SVC(**hp))
                  ]
        p = Pipeline(steps=steps)        
        
        return p
    
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
        
        model2 = CreditDefault_ANN()
        model2.fit_model(X_train2=X_train2, y_train2=y_train2)
    

        
    
class CreditDefault_SVCLinear(CreditDefault_SVCBase):
    def make_model(self, hp):
        steps = [ #('feature_selection', SelectFromModel(DecisionTreeClassifier(ccp_alpha=0.001225))),
            ('scale', StandardScaler()),  
                  ('model', SVC(**hp))
                  ]
        p = Pipeline(steps=steps)        
        
        return p
    
    
    def get_init_params(self):
        return    { 'kernel': 'poly', 'C': 5.0,  'coef0':2 , 'degree':3, 'gamma':0.003}

    
    def get_cv_params(self):
        return {
            'model__C': np.logspace(np.log10(0.5), np.log10(20), 5)
            #'model__C': np.linspace(0.5, 5, 8),
            #'model__gamma': np.logspace(np.log10(1e-3), np.log10(0.01), 5),
            #'model__coef0': np.linspace(4, 6, 3),

            #'model__C': np.linspace(0.4, 1.5, 20),
        }
    def get_abbrev_name(self):
        return 'svm-linear-cd'
    
    def get_long_name(self):
        return 'SVM(Linear) CD'
    

class CreditDefault_SVCRBF(CreditDefault_SVCBase):
    
    def get_init_params(self):
        return    {
            'kernel': 'rbf',
            'C': 1.0,
        }
   
    def get_cv_params(self):
        return {
            #'C': np.logspace(np.log10(1/2000), np.log10(0.05), 20)
            'model__C': np.linspace(0.5, 50, 5),
            'model__gamma': np.logspace(np.log10(1e-3), np.log10(0.005), 10)
        }

        
    def get_abbrev_name(self):
        return 'svm-rbf-cd'
    
    def get_long_name(self):
        return 'SVM(RBF)-cd'
    
class CreditDefault_DT(CreditDefaultBase):
    def __init__(self):
        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "CreditDefault"
        
    def get_abbrev_name(self):
        return  'dt-cd'
    
    def get_long_name(self):
        return 'Decision Tree (CD)'
    

    def make_model(self, hp):
        return DecisionTreeClassifier(**hp)
    
    
    def get_init_params(self):
        return    {'ccp_alpha': 0.01}
   
    def get_cv_params(self):
        return {
            'ccp_alpha': np.linspace(0.0, 0.06, 50)
        }

class CreditDefault_ANN(CreditDefaultBase):
    def __init__(self, hidden_layer_sizes=(8,)):
        self.hidden_layer_sizes = hidden_layer_sizes
        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "CreditDefault"
    
    def preprocess(self, X_train, y_train, X_test, y_test):
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)
        
        X_train = (X_train - self.mean) / self.std
        X_test = (X_test - self.mean) / self.std
        
        return X_train, y_train, X_test, y_test
    
    
    def make_model(self, hp):
        return MLPClassifier(**hp, max_iter=200)
        
    
    def get_cv_class(self):
        return RandomizedSearchCV2    
    
    
    def get_init_params(self):
        return    {'hidden_layer_sizes': self.hidden_layer_sizes, 
                   'batch_size': 64, 
                   'learning_rate_init':0.001,
                   'momentum': 1,
                   'solver':'sgd'}    
    
    def get_cv_params(self):
        return {'learning_rate_init': np.logspace(np.log10(0.0001), np.log10(0.02), 10),
                'momentum': [0.9, 0.99, 0.999],
                'alpha' : np.linspace(0.0, 0.02, 10)
                }
    
class CreditDefault_AdaBoost(CreditDefaultBase, BoostingVCMixin):
    def __init__(self):

        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "CreditDefault"
    
    def get_abbrev_name(self):
        return  'boost-cd'
    
    def get_long_name(self):
        return 'AdaBoost-cd'
    

    def make_model(self, hp):
        return AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=5),
            **hp
            )
    def get_cv_class(self):
        return BoostSearchCV
    
    
    def get_init_params(self):
        return    {'learning_rate': 1.0, 'n_estimators':200}
   
    def get_cv_params(self):
        #return {'n_estimators': np.arange(1, 6), 
                #"base_estimator__max_depth": [1, 2,3]}
        
        return {'n_estimators': np.arange(1, 100), 
                "base_estimator__max_depth": [1,2, 3, 4, 5, 6, 7, 8],
                }
    
            

    

if __name__=='__main__':
    now = str(datetime.datetime.now()).replace(':','')
    setup_logging('log/' + now + '.log')
    
    if 1:
        model = CreditDefault_kNN(False)
        X_train, y_train, X_test, y_test = model.get_data()
        
       
        model.fit_model(5, params={'model__n_neighbors':8})
        model.eval_performance()
        model.make_learning_curve(filename='knn-learn-ses.pdf', title='kNN Learning Curve on Credit Default data', ylabel='F1 Macro')
        
        model.make_validation_curve('model__n_neighbors', 'knn-val-ses.pdf', title='kNN Model Complexity Curve on Credit Default data', 
                                    xlabel='Neighbors',   ylabel='F1 Macro')        
    
        plt.close('all')
    if 0:
        model = CreditDefault_kNN(True)
        X_train, y_train, X_test, y_test = model.get_data()
        
       
        model.fit_model(4)
        model.eval_performance()
        model.make_learning_curve(filename='knn-learn-ses-down.pdf', title='kNN Learning Curve on CreditDefault data', ylabel='F1 Macro')
        
        model.make_validation_curve('model__n_neighbors', 'knn-val-ses-down.pdf', title='kNN Model Complexity Curve on CreditDefault data', 
                                    xlabel='Neighbors',   ylabel='F1 Macro')        
    
        plt.close('all')
        
    if 1:
        model = CreditDefault_SVCLinear()
        model.fit_model(5)
        model.make_learning_curve(filename='svc-learn-ploy-ses.pdf', title='SVM(Linear) Learning Curve on Credit Default Data', ylabel='F1-Macro Score')
        model.eval_performance()
        
        model.make_validation_curve('model__C', 'svc-val-poly-ses.pdf', title='F1-Macro Score vs C', 
                                    xlabel='C', ylabel='F1-Macro Score')
        #model.make_validation_curve('model__coef0', 'svc-val2-poly-ses.pdf', title='F1-Macro Score vs $c_0$', 
        #                            xlabel='$c_0$', ylabel='F1-Macro Score')        
        
        #model.fit_model(5, params={'model__C': 6.2})
        #model.eval_performance()
        #model.make_learning_curve(filename='svc-learn-ploy-ses.pdf', title='SVM(Linear) Learning Curve on Credit Default Data', cv=8,ylabel='F1-Macro Score')
        
        
        plt.close('all')
    
    if 1:
        model = CreditDefault_SVCRBF()
        model.fit_model(5)
        #model.eval_performance()
        #model.show_sv()
        #model.test_sv()
        
        #model.make_validation_curve('model__C', 'svc-val-rbf-ses.pdf', title='SVM(RBF) Model Complexity Curve on Credit Default Data', 
        #                            xlabel='C', ylabel='F1-Macro Score')
        #model.make_validation_curve('model__gamma', 'svc-val-rbf2-ses.pdf', title='SVM(RBF) Model Complexity Curve on Credit Default Data', 
        #                            xlabel='$\gamma$', ylabel='F1-Macro Score')

        model.fit_model(5, params={'model__C': 20})
        model.eval_performance()
        
        
        model.make_learning_curve(filename='svc-learn-rbf-ses.pdf', title='SVM(RBF) Learning Curve on Credit Default Data', ylabel='F1-Macro Score', 
                                  cv=7) #8 is best
        
        
        plt.close('all')
        
    if 1:
        model = CreditDefault_DT()
        
        model.fit_model(5)
        #with open('tree.txt', 'w') as fh:
            #print(tree.export_text(model.trained_model['cv'].best_estimator_), file=fh)
        
        #X_train, y_train, X_test, y_test = model.get_data()
        #clf = DecisionTreeClassifier(ccp_alpha=0.01)
        #clf.fit(X_train, y_train)
        #with open('tree2.txt', 'w') as fh:
            #print(tree.export_text(clf), file=fh)
        
        #sklearn.metrics.f1_score(y_test, clf.predict(X_test))
            
        model.eval_performance()
        model.make_learning_curve(filename='dt-learn-ses.pdf', title='Decision Tree Learning Curve on USPS Data')
        model.make_validation_curve('ccp_alpha', 'dt-val-ses.pdf', 'Decision Tree Validation Curve on USPS data', xlabel='$\\alpha$')
    
        plt.close('all')
        
    if 1:
        model = CreditDefault_AdaBoost()
        model.fit_model(5)
        model.eval_performance()
        model.make_learning_curve(filename='boost-learn-cd.pdf', title='AdaBoost Learning Curve on Credit Default Data')
        
        rs, ts, vs = model.make_validation_curve('n_estimators', 'boost-val-cd.pdf', 'AdaBoost Validation Curve on Credit Default Data', 
                                                 xlabel='Weak Learners', rng=np.arange(1, 200))    
        idx = min(np.nonzero(vs<=vs.min()+1e-4)[0])
        
        #logging.info('Best number of learners if we scrafice 0.1% performance: {}'.format(rs[idx]))
        #n_estimators = rs[idx]
        #estimator = model.trained_model['cv'].best_estimator_
        #estimator.set_params(n_estimators=n_estimators)
        #model.eval_performance(estimator, reestimate=True)


        
    
    if 1:
        model = CreditDefault_ANN(hidden_layer_sizes=(8,))
        X_train, y_train, X_test, y_test = model.get_data()
        X_train, y_train, X_test, y_test =  model.preprocess(X_train, y_train, X_test, y_test)
        
        for mb_size in [32, 64, 128, 256]:
            model.model.set_params(batch_size=mb_size, solver='sgd')
            make_time_curves(model.model, X_train, y_train, 
                             {'learning_rate_init':[1e-4, 1e-3, 1e-2, 1e-1, 0.2]}, 
                             name_formatters={'batch_size':'mb_sz', 'learning_rate_init':'lr'},
                             title='Learning Curve with Mini-batch Size {}'.format(mb_size),
                             max_epoch=100, filename='ann-lr_{}-ses.pdf'.format(mb_size)
                             , is_loss=True)
    if 1:
        model = CreditDefault_ANN()
        model.fit_model(5)
        model.eval_performance()
        model.make_learning_curve(filename='ann-learn-cd.pdf', title='ANN Learning Curve on Credit Default Data')
        model.make_validation_curve("alpha", "ann-val-cd.pdf", "Error vs $\\alpha$", xlabel='$\\alpha$')
        
        model.fit_model(5, params={'max_iter':120})
        model.eval_performance()
    
    if 0: # not needed
        model = CreditDefault_ANN(hidden_layer_sizes=(10,))
        #model.fit_model(5)
        #model.eval_performance()
        
        X_train, y_train, X_test, y_test = model.get_data()
        X_train, y_train, X_test, y_test =  model.preprocess(X_train, y_train, X_test, y_test)
        
        
        for arch, alpha in zip([(2,),  (4,), (8,),  (10,), (20,)], [0.5, 0.1, 0.07, 0.07]):
                model.model.set_params(batch_size=64, solver='sgd', learning_rate_init= 0.01,
                                       hidden_layer_sizes=arch, max_iter=150, alpha=alpha)
                make_time_curves(model.model, X_train, y_train, 
                             #{'hidden_layer_sizes':[(10,), (20,), (50,), (100,), (50,50), (100, 100)]}, 
                             {'hidden_layer_sizes':[arch]}, 
                             title='Learning Curve for Arch=[{}]'.format(",".join(map(str, arch))),
                             max_epoch=200, filename='ann-lr-arch-{}-ses.pdf'.format("-".join(map(str,arch))), 
                             validation=5, is_loss=False,
                             #cv_params={'alpha':np.logspace(np.log10(0.01), np.log10(1), 10)},
                             name_formatters={'hidden_layer_sizes':''},
                             formatter={'hidden_layer_sizes': lambda x: ""},
                             same_color=False,
                             train_label='training', cv_label='validation', ylabel="F1-Macro Score", scoring=model.get_scorer_name(),
                             ylim=[0.64, 0.72]
                             )
    if 1:
        model = CreditDefault_ANN(hidden_layer_sizes=(8,))    
        model.fit_model(5, params=dict(hidden_layer_sizes=(8,), alpha=0.000))
        model.make_validation_curve("hidden_layer_sizes", "ann-hidden-ses.pdf", title="Effect of Hidden Units", xlabel='Hidden Units',
                                    rng=[(x,) for x in range(2, 16)], ylabel='F1-Macro Score', shuffle=True, rounds=10, random_state=8888)
        
        
    if 1:
        
        model = CreditDefault_ANN(hidden_layer_sizes=(6,))    
        model.fit_model(5, params=dict(hidden_layer_sizes=(6,), alpha=0.0089))  
        model.make_learning_curve(filename='ann-learn-ses.pdf', title='ANN Learning Curve on Credit Default Data', ylabel='F1-Macro Score',
                                  rounds=10, random_state=8888)
    
    if 1:
        #model = CreditDefault_ANN(hidden_layer_sizes=(6,))
        #model.fit_model(5)
        with open('ann-cd.pkl', 'rb') as fh:
            model = pkl.load(fh)    
            
        
        
        X_train, y_train, X_test, y_test = model.get_data()
        X_train, y_train, X_test, y_test =  model.preprocess(X_train, y_train, X_test, y_test)
        
        est = sklearn.base.clone(model.trained_model['cv'].best_estimator_)
        make_time_curves(est, X_train, y_train, 
                     {}, 
                     title='F1-Macro Score vs Epochs',
                     max_epoch=400, filename='ann-lr-ep-ses.pdf', 
                     validation=10, is_loss=False,
                     #cv_params={'alpha':np.logspace(np.log10(0.01), np.log10(1), 10)},
                     name_formatters={'hidden_layer_sizes':''},
                     formatter={'hidden_layer_sizes': lambda x: ""},
                     same_color=False,
                     train_label='training', cv_label='validation', ylabel="F1-Macro Score", scoring=model.get_scorer_name(),
                     ylim=[0.64, 0.72]
                     )
        
        