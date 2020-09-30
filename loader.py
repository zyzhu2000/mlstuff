import numpy as np
import pandas as pd
import logging
import arff

''' 
THIS FILE PROVIDES A UNIFORM INTERFACE TO ACCESS DATA 
'''

def make_one_hot_encoding(df, columns, drop=True, prefix=''):
    if prefix=='':
        prefix = columns
    df = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype('category')
        
    dummies = pd.get_dummies(df[columns], prefix=prefix)
    df2 = pd.concat([df, dummies], axis=1)
    if drop:
        df3 = df2.drop(columns=columns)
    return df3
    
def make_label_encoding(df, columns, drop=True):    
    df = df.copy()
    for column in columns:
        s = df[column].astype('category')
        df[column] = s.cat.codes
    return df
        
    

class Data(object):
    def __init__(self):
        self.n_classes = 0
        self.X = None
        self.y = None
    
    
class OptDigits(object):
    def load(self, name, params):
        assert name=='OptDigits'    
        df1 = pd.read_csv('data/optdigits.tra', header=None)
        df2 = pd.read_csv('data/optdigits.tes', header=None)
        
        df = pd.concat([df1, df2])
        x = df.values
        
        data = Data()
        data.X = x[:,:-1]
        data.y = x[:,-1]
        data.n_classes = 10
        
        return data

class SpamLoader(object):
    def load(self, name, params):
        assert name=='spam'    
        df = pd.read_csv('data/spambase.data', header=None)
        
        data = Data()
        data.X = df.iloc[:,:-1].values
        data.y = df.iloc[:,-1].values
        data.n_classes = 2
        
        return data
    
class USPSLoader(object):
    # https://web.stanford.edu/~hastie/StatLearnSparsity_files/DATA/zipcode.html
    
    def add_noise(self, X, pct):
        np.random.seed(999)
        i_noise = np.random.choice(X.shape[1], size=(X.shape[0], int(pct*X.shape[1])), replace=True)
        noise = np.random.uniform(X.min()*1.5, X.max()*1.5, size=i_noise.shape)
        X2 = X.copy()
        for i in range(len(X)):
            X2[i, i_noise[i,:]] += noise[i,:]
        return X2
        
        
        
    def load(self, name, params):
        df1 = pd.read_csv('data/zip.train', header=None, sep=' ')
        df2 = pd.read_csv('data/zip.test', header=None, sep=' ')
        
        #df = pd.concat([df1, df2])
        data = Data()
        #data.X = df.iloc[:,1:-1].values
        #data.y = df.iloc[:,0].values.astype(np.int64)
        data.X_train = df1.iloc[:,1:-1].values
        data.y_train = df1.iloc[:,0].values.astype(np.int64)
        
        data.X_test = df2.iloc[:,1:].values
        data.y_test = df2.iloc[:,0].values.astype(np.int64)
        
        data.X_train = self.add_noise(data.X_train, 0.3)
        data.X_test = self.add_noise(data.X_test, 0.3)
        
        #add noise
        
        
        data.n_classes = 10
        
        return data
    
class SeismicLoader(object):
    def load(self, name, params):
        assert name=='seismic'    
        with open('data/seismic-bumps.arff') as fh:
            dataset = arff.load(fh)
        
        columns = [x[0] for x in dataset['attributes']]
        df = pd.DataFrame(dataset['data'], columns=columns)
        df = df.sample(frac=1, random_state=0)
        
        if params is not None and params['encoding']=='onehot':
            df = make_one_hot_encoding(df, ['seismoacoustic'])
            df = make_label_encoding(df, ['seismic', 'shift', 'ghazard'])
        else:
            df = make_label_encoding(df, ['seismic', 'shift', 'seismoacoustic', 'ghazard'])
        df['class'] = df['class'].astype(int)
            
        data = Data()
        data.y = df['class'].values
        df = df.drop(columns=['class'])
        data.X = df.iloc[:,:-1].values
        
        data.n_classes = 2
        
        return data

class CreditDefaultLoader(object):
    def load(self, name, params):
        assert name=='CreditDefault'    
        df = pd.read_csv('data/credit_default.csv', index_col='ID')
        df = df.sample(frac=1, random_state=0)
        df = make_one_hot_encoding(df, ['EDUCATION'])
            
        data = Data()
        data.y = df['default payment next month'].values[:15000]
        df = df.drop(columns=['default payment next month'])
        data.X = df.iloc[:15000,:].values
        
        data.n_classes = 2
        
        return data


class DataLoader(object):
    loaders = {
        'OptDigits': OptDigits,
        'spam': SpamLoader,
        'USPS': USPSLoader,
        'seismic': SeismicLoader,
        'CreditDefault': CreditDefaultLoader
        
    }
    def load(self, name, params=None):
        loader = self.loaders[name]()
        ret = loader.load(name, params)
        return ret
    
if __name__=='__main__':
    loader = DataLoader()
    # loader.load('USPS')
    #loader.load('seismic', {'encoding':'onehot'})
    loader.load('CreditDefault')
    
        
    
    
