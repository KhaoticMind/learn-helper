# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:23:31 2015

@author: ur57
"""

def modelSearch(view, data, y, classifiers, cv=3, shuffle=False, random_state=None):
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
                    t = view.apply(doProcess, local_clf, label_clf, X, y, train_index, test_index, label_dados)
                    tasks.append(t)
    return tasks

def doProcess(clf, label_clf, X, y, train_index, test_index, label_dados):
    clf.fit(X[train_index], y[train_index])
    train_score = clf.score(X[train_index], y[train_index])
    test_score = clf.score(X[test_index], y[test_index])
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
