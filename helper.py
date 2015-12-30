# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:23:31 2015

@author: ur57
"""


def data_load(label='train'):
    from sklearn.externals import joblib
    train = joblib.load(label + '_train.pkl')
    test = joblib.load(label + '_test.pkl')

    return train, test


def data_persist(X, label='train', test_size=0.25):
    from sklearn.externals import joblib
    from sklearn.cross_validation import ShuffleSplit
    from pandas import DataFrame

    folds = ShuffleSplit(X.shape[0], 1, test_size=test_size)
    train, test = list(folds)[0]
    if isinstance(X, DataFrame):
        joblib.dump(X.iloc[train], label + '_train.pkl')
        joblib.dump(X.iloc[test], label + '_test.pkl')
    else:
        joblib.dump(X[train], label + '_train.pkl')
        joblib.dump(X[test], label + '_test.pkl')


def printProgress(tasks):
    results = getResults(tasks)

    res = results.groupby(['label_clf', 'label_dados', 'params'],
                          as_index=False).mean()
    res_idxs = res.groupby(['label_clf', 'label_dados'],
                           as_index=False)['test_score'].idxmax()
    res = res.iloc[res_idxs.values, ]

    n_done = results.shape[0]
    pct_done = (n_done / len(tasks)) * 100

    print("{:d} of {:d} ({:.2f}%)".format(n_done, len(tasks), pct_done))
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
    one_engine_by_host = dict((hostname, engine_id)
                              for engine_id, hostname in hostnames.items())
    one_engine_by_host_ids = list(one_engine_by_host.values())
    return client[one_engine_by_host_ids]


def modelSearch(view, data_label, y_label, n_samples, classifiers, cv=3, shuffle=False, random_state=None, metric='acc'):
    '''
    data: Uma lista no formato de tuplas (label_dado, dado)
    classifiers: Uma lista no formado (label_clf, clf, params_dic)
    cv: Numero de cross validations para executar para cada dado/clf/parametro
    '''
    from sklearn.cross_validation import KFold
    from sklearn.grid_search import ParameterGrid
    from copy import copy

    tasks = []
    for label_dados in data_label:
        folds = KFold(n_samples, cv,
                      shuffle=shuffle, random_state=random_state)
        for train_index, test_index in folds:
            for label_clf, clf, params_dict, fit_params_dict in classifiers:
                params = ParameterGrid(params_dict)
                for param in params:
                    local_clf = copy(clf)
                    local_clf.set_params(**param)
                    t = view.apply(doProcess, local_clf, label_clf,
                                   label_dados, y_label, train_index,
                                   test_index, metric, param, fit_params_dict)
                    tasks.append(t)
    return tasks


def doProcess(clf, label_clf, label_dados, y_label, train_index, test_index,
              metric='acc', params_dict=None, fit_params_dict=None):
    from sklearn.externals import joblib
    from sklearn.metrics import accuracy_score, roc_auc_score

    X = joblib.load(label_dados + '.pkl', mmap_mode='r+')
    y = joblib.load('y_values.pkl', mmap_mode='r+')

    y_train = y[train_index]
    y_test = y[test_index]

    if fit_params_dict is None:
        clf.fit(X[train_index], y_train)
    else:
        clf.fit(X[train_index], y_train, **fit_params_dict)

    y_train_pred = clf.predict(X[train_index])
    y_test_pred = clf.predict(X[test_index])

    if metric == 'auc':
        train_score = roc_auc_score(y_train, y_train_pred)
        test_score = roc_auc_score(y_test, y_test_pred)
    else:  # metric == 'acc':
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)

    if params_dict is None:
        params = clf.get_params()
    else:
        params = params_dict

    params = frozenset(zip(params.keys(), params.values()))
    return (label_clf, params, label_dados, train_score, test_score)

def getResults(tasks):
    import pandas as pd
    res = []
    for t in tasks:
        try:
            if t.ready():
                res.append(t.get())
        except:
            pass

    return pd.DataFrame(res, columns = ['label_clf', 'params', 'label_dados', 'train_score', 'test_score'])
