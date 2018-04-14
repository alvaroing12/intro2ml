import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt
from numpy.core.multiarray import dtype


def parsedata(fileTrain, feature_size):

    train = pd.read_csv(fileTrain)
    ids = train.ix[:,0]
    Y = train.ix[:,1].as_matrix()
    X = train.ix[:,2:(feature_size+2)].as_matrix()

    return ids, Y, X

def buildmodel(X_train,X_test,Y_train,Y_test,lambda_coeff):

    # Fit the model

    A = np.dot(X_train.T, X_train)
    B = lambda_coeff * np.identity(n = X_train.shape[1], dtype=np.float64)
    inverse = np.linalg.pinv(A + B)
    w_hat = np.dot(inverse , np.dot(X_train.T, Y_train))
    Y_pred = np.dot(X_test, w_hat.T)
    rmse = (mean_squared_error(Y_test, Y_pred))**0.5

    return rmse

def formoutput(arr):

    output = pd.DataFrame({'score': arr})
    output.to_csv('output.csv', index=False, header=False)
