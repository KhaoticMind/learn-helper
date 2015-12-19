# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:23:31 2015

@author: ur57
"""


def printProgress(tasks):
    import numpy as np
        
    results = getResults(tasks)
    
    res = results.groupby(['label_clf', 'label_dados', 'params'], as_index=False).mean()
    res = results.groupby(['label_clf', 'label_dados'], as_index=False).max()
    
    done = [t.ready() for t in tasks]
    pct_done = np.mean(done) * 100
    n_done  = np.sum(done)
    print("{:d} of {:d} ({:.2f}%)".format(n_done, len(tasks), pct_done) )
    print(res)

def persistData(X, label_dados):
    from sklearn.externals import joblib
        
    filename = label_dados + '.pkl'
    joblib.dump(X, filename, compress=0)

def hostname():
    """Return the name of the host where the function is being called"""
    import socket
    return socket.gethostname()
    
def applyPerHost(directview, func, *args, **kwargs):
    perhost = oneEnginePerHost(directview)
    perhost.apply_sync(func, *args, **kwargs)
    
def oneEnginePerHost(client):
    from time import sleep
    
    directview = client.direct_view()
    res = directview.apply(hostname)

    while not res.ready():
        sleep(0.2)
    
    hostnames = res.get_dict()
    one_engine_by_host = dict((hostname, engine_id) for engine_id, hostname
                      in hostnames.items())
    one_engine_by_host_ids = list(one_engine_by_host.values())
    return client[one_engine_by_host_ids]

def modelSearch(view, data, y, classifiers, cv=3, shuffle=False, random_state=None, metric='acc'):
    '''
    data: Uma lista no formato de tuplas (label_dado, dado)
    classifiers: Uma lista no formado (label_clf, clf, params_dic)
    cv: Numero de cross validations para executar para cada dado/clf/parametro
    '''
    from sklearn.cross_validation import KFold
    from sklearn.grid_search import ParameterGrid
    from copy import copy
    
    tasks = []
    for label_dados, X in data:
        folds = KFold(X.shape[0], cv, shuffle, random_state)
        for train_index, test_index in folds:
            for label_clf, clf, params_dic in classifiers:
                params = ParameterGrid(params_dic)
                for param in params:                    
                    local_clf = copy(clf)
                    local_clf.set_params(**param)
                    t = view.apply(doProcess, local_clf, label_clf, label_dados, y, train_index, test_index, metric)
                    tasks.append(t)
    return tasks

def doProcess(clf, label_clf, label_dados, y, train_index, test_index, metric='acc'):
    from sklearn.externals import joblib
    from sklearn.metrics import  accuracy_score, roc_auc_score
    X = joblib.load(label_dados + '.pkl', mmap_mode='r+')

    y_train = y[train_index]
    y_test =  y[test_index]            
    clf.fit(X[train_index], y_train)
    
    y_train_pred = clf.predict(X[train_index])
    y_test_pred = clf.predict(X[test_index])
    
    if metric == 'auc':
        train_score = roc_auc_score(y_train, y_train_pred)
        test_score =  roc_auc_score(y_test, y_test_pred)
    else: # metric == 'acc':
        train_score = accuracy_score(y_train, y_train_pred)
        test_score =  accuracy_score(y_test, y_test_pred)
        
    pars = clf.get_params()
    pars = frozenset(zip(pars.keys(), pars.values()))
    return (label_clf, pars, label_dados, train_score, test_score)

def getResults(tasks):
    import pandas as pd
    res = []
    for t in tasks:
        if t.ready():
            res.append(t.get())
            
    return pd.DataFrame(res, columns = ['label_clf', 'params', 'label_dados', 'train_score', 'test_score'])
