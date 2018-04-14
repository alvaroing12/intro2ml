import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt
from numpy.core.multiarray import dtype
from sklearn.preprocessing import RobustScaler


def parsedata(fileTrain, feature_size, y_label_present=True):

    data = pd.read_csv(fileTrain)
    
    if y_label_present:
        aux = data.ix[:,:].as_matrix()
        np.random.shuffle(aux)
        ids = aux[:,0]
        Y = aux[:,1]
        X = aux[:,2:(feature_size+2)]
    else:
        ids = data.ix[:,0]
        Y = -1
        X = data.ix[:,1:(feature_size+1)].as_matrix()

    robust_scaler = preprocessing.RobustScaler()
    X_scaled = robust_scaler.fit_transform(X)
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

def formoutput(ids, data, filename, use_header=False):

    output = pd.DataFrame({'Id':ids,'y': data})
    output.to_csv(filename, index=False, header=use_header)
