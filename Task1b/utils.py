'''

TASK 1 B

'''

import pandas as pd
from sklearn.linear_model import Ridge
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

def buildmodel(X_train,X_test,Y_train,Y_test,lambda_coeff):

    # Fit the model

    A = np.dot(X_train.T, X_train)
    B = lambda_coeff * np.identity(n = X_train.shape[1], dtype=np.float64)
    inverse = np.linalg.pinv(A + B)
    w_hat = np.dot(inverse , np.dot(X_train.T, Y_train))
    Y_pred = np.dot(X_test, w_hat.T)
    rmse = (mean_squared_error(Y_test, Y_pred))**0.5

    return rmse

def build_baseline(X,Y):
    w_hat = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    Y_pred = np.dot(X, w_hat.T)
    rmse = (mean_squared_error(Y, Y_pred))**0.5
    return w_hat, rmse    

def formoutput(arr):

    output = pd.DataFrame({'w_hat': arr})
    output.to_csv('output.csv', index=False, header=False)
