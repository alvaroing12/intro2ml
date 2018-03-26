'''

TASK 1 B

'''

import pandas as pd
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt
from numpy.core.multiarray import dtype


def parsedata(fileTrain):

    train = pd.read_csv(fileTrain)
    ids = train.ix[:,0]
    Y = train.ix[:,1].as_matrix()
    X = train.ix[:,2:7].as_matrix()

    return ids, Y, X

def transform_features(X):
    linear = X
    quad = np.square(X)
    exp = np.exp(X)
    cos = np.cos(X)
    const = np.ones(shape=(X.shape[0],1), dtype=np.float64)
    return np.concatenate((linear,quad,exp,cos,const), axis=1)

def build_baseline_v4(X_train,X_test,Y_train,Y_test,lambda_coeff):
    elastic = ElasticNetCV(alpha=lambda_coeff, l1_ratio = 0.5, normalize=False, max_iter=1e5, fit_intercept=True)
    elastic.fit(X_train,Y_train)
    
    
    w_hat = elastic.coef_
    Y_pred = elastic.predict(X_test)
    rmse = (mean_squared_error(Y_test, Y_pred))**0.5
    return w_hat, rmse

def build_baseline_v5(X,Y):
    regr = ElasticNet(alpha=0.28, l1_ratio=0.62, copy_X = True, normalize=True, max_iter=1e5, random_state=0, fit_intercept=False)
    regr.fit(X,Y)
    
    alpha = 0
    l1_ratio = 0
    w_hat = regr.coef_

    Y_pred = regr.predict(X)
    rmse = (mean_squared_error(Y, Y_pred))**0.5
    return w_hat, rmse, alpha, l1_ratio, regr

def build_baseline_v3(X_train,X_test,Y_train,Y_test,lambda_coeff):

    # Fit the model

    A = np.dot(X_train.T, X_train)
    B = lambda_coeff * np.identity(n = X_train.shape[1], dtype=np.float64)
    inverse = np.linalg.pinv(A + B)
    w_hat = np.dot(inverse , np.dot(X_train.T, Y_train))
    Y_pred = np.dot(X_test, w_hat.T)
    rmse = (mean_squared_error(Y_test, Y_pred))**0.5

    return w_hat, rmse

def build_baseline_v2(X_train, X_test, Y_train, Y_test):
    w_hat = np.dot(np.linalg.pinv(np.dot(X_train.T, X_train)), np.dot(X_train.T, Y_train))
    Y_pred = np.dot(X_test, w_hat.T)
    rmse = (mean_squared_error(Y_test, Y_pred))**0.5
    return w_hat, rmse

def build_baseline(X,Y):
    w_hat = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    Y_pred = np.dot(X, w_hat.T)
    rmse = (mean_squared_error(Y, Y_pred))**0.5
    return w_hat, rmse    

def formoutput(arr):

    output = pd.DataFrame({'w_hat': arr})
    output.to_csv('output.csv', index=False, header=False)
