from  classifiers import *
import tqdm
import itertools
from joblib import Parallel, delayed
from sklearn.model_selection import RandomizedSearchCV
import sklearn.base

class RandomizedSearchCV2(RandomizedSearchCV):
    def __init__(self, estimator, grid, **kwargs):
        super().__init__(estimator, grid, n_iter=50, **kwargs)
    

class USPS_ANN(USPSBase):
    def __init__(self, hidden_layer_sizes=(50,), b_data_aug=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        hp = self.get_init_params()
        self.model = self.make_model(hp)
        self.data_name = "USPS"
        self.b_data_aug = b_data_aug

    def rotate(self, X, size):
        image = X.reshape((16,16))
        img_pil = Image.fromarray(image)

        degree = np.random.uniform(-3, 3)
        img_pil2 = img_pil.rotate(degree)
                  
        
        resized_image = np.asarray(img_pil2)  
       
        X2 = resized_image.reshape(-1)
        return X2
    
    def add_noise(self, X, pct):
        np.random.seed(888)
        i_noise = np.random.choice(X.shape[1], size=(X.shape[0], int(pct*X.shape[1])), replace=True)
        noise = np.random.uniform(X.min()*1.5, X.max()*1.5, size=i_noise.shape)
        X2 = X.copy()
        for i in range(len(X)):
            X2[i, i_noise[i,:]] += noise[i,:]
        return X2
    
        
        
    def preprocess(self, X_train, y_train, X_test, y_test):
        if not hasattr(self, 'b_data_aug') or not self.b_data_aug:
            return X_train, y_train, X_test, y_test
        
        m = 16
        n = X_train.shape[0]
        X_train2 = np.zeros((n, m*m), dtype=bool)
        for i in range(n):
            X_train2[i, :] = self.rotate(X_train[i, :], m)
        
        
        X_train3 = np.concatenate((X_train, X_train2))
        
        y_train3 = np.concatenate((y_train, y_train))
        
        return X_train3, y_train3, X_test, y_test
    
    
    def make_model(self, hp):
        return MLPClassifier(**hp, max_iter=400)
    
    def get_cv_class(self):
        return RandomizedSearchCV2    
    
    
    def get_init_params(self):
        return    {'hidden_layer_sizes': self.hidden_layer_sizes, 
                   'batch_size': 128, 
                   'learning_rate_init':0.01}    
    
    def get_cv_params(self):
        return {'learning_rate_init': np.logspace(np.log10(0.001), np.log10(0.02), 5),
                'momentum': 1 - np.logspace(np.log10(1e-8), np.log10(0.1), 10),
                'hidden_layer_sizes': [(80,), (100,), (120,)],
                'alpha' : np.linspace(0.01, 0.02, 10)
                }
    


def make_one_time_curve(estimator, X_train, y_train, X_test, y_test, n_classes,  max_epoch, scoring):
    training_error = np.zeros(max_epoch)
    dev_error = np.zeros(max_epoch)
    scorer = sklearn.metrics.get_scorer(scoring)     
    
    for epoch in range(max_epoch):
        estimator.partial_fit(X_train, y_train, n_classes)
        training_error[epoch] = scorer(estimator, X_train, y_train)
        if X_test is not None:
            dev_error[epoch] = scorer(estimator, X_test, y_test)

    return training_error, dev_error, estimator.loss_curve_


def make_time_curves(estimator, X, y, param_grid, max_epoch, validation=None, 
                     title='', filename='base.pdf', formatter={}, 
                     name_formatters={},
                     n_jobs=-1, verbose=1, is_loss=True, cv_params=None, same_color=True,
                     train_label='', cv_label='', ylabel="Error", scoring='accuracy', xlim=None, ylim=None, flip=False):
    kf = None
    if validation is not None:
        if validation<1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation, random_state=8888)
        else:
            kf = StratifiedKFold(n_splits=validation)
    else:
        X_train = X
        y_train = y
        X_test = y_test = None
   
    
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    n_classes = np.unique(y)
    
    estimators = []
    legends = []
    for params in  itertools.product(*values):
        d = dict(zip(keys, params))
        est = sklearn.base.clone(estimator)
        est.set_params(**d)
        if cv_params is not None:
            cv = GridSearchCV(est, cv_params, n_jobs=-1)
            cv.fit(X_train, y_train)
            logging.info("{} : {}".format(params, cv.best_params_))
            est = cv.best_estimator_
        estimators.append(est)
        
        legend = []
        for p, v in d.items():
            if p in formatter:
                s = formatter[p](v)
            else:
                s = v
            if p in name_formatters:
                name = name_formatters[p]
            else:
                name = p
            if name and s:
                legend.append("{}={}".format(name, s))
            elif name:
                legend.append(name)
            elif s:
                legend.append(s)
            else:
                legend.append('')
                
                
        legends.append(",".join(legend))
    
    if kf is None:
            
        #n_jobs = 1
        res = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                    delayed(make_one_time_curve)(sklearn.base.clone(est), X_train, y_train, X_test, y_test, n_classes, max_epoch, scoring) 
                    for est in estimators
            )
    else:
        terr_  = np.zeros((kf.get_n_splits(), len(estimators), max_epoch))
        derr_  = np.zeros((kf.get_n_splits(), len(estimators), max_epoch))
        loss_  = np.zeros((kf.get_n_splits(), len(estimators), max_epoch))
        
        for k, (idx_train, idx_test) in enumerate(kf.split(X, y)):
            X_train = X[idx_train,:]
            y_train = y[idx_train]
            X_test = X[idx_test,:]
            y_test = y[idx_test]
            
            res = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                        delayed(make_one_time_curve)(sklearn.base.clone(est), X_train, y_train, X_test, y_test, n_classes, max_epoch, scoring) 
                        for est in estimators
                )            
            for i, (training_err, dev_err, loss) in enumerate(res):
                terr_[k, i, :] = training_err
                derr_[k, i, :] = dev_err
                loss_[k, i, :] =  loss
        
        res = []
        for i in range(len(estimators)):
            res.append([terr_[:, i, :].mean(axis=0), derr_[:, i, :].mean(axis=0), loss_[:, i, :].mean(axis=0)])
    
    plt.figure()
    x = np.arange(1, max_epoch+1)
    for i, (training_err, dev_err, loss) in enumerate(res):
        if is_loss:
            p = plt.plot(x, loss, label= legends[i], linewidth=1)
        else:
            if flip:
                p = plt.plot(x, 1-training_err, label= train_label + legends[i], linewidth=1)
            else:
                p = plt.plot(x, training_err, label= train_label + legends[i], linewidth=1)
        
        if validation is not None and not is_loss:
            kw = {}
            if same_color:
                kw['color'] = p[0].get_color()
            
            if flip:
                plt.plot(x, 1-dev_err, label= cv_label + legends[i], linewidth=1, linestyle='dashed',  **kw)
            else:
                plt.plot(x, dev_err, label= cv_label + legends[i], linewidth=1, linestyle='dashed',  **kw)
            
    plt.legend()
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    if not is_loss:
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    else:
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    fontsize = 12
    ax = plt.gca()
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')    
        
        
    plt.grid(True)
    plt.title(title, fontsize=16)
    plt.savefig(filename)
    plt.close()
    
        
