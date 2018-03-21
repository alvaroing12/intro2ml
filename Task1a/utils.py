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
    X = train.ix[:,2:11].as_matrix()

    return ids, Y, X

def buildmodel(train_data,test_data,target_train,target_test,lambda_coeff):

    # Fit the model

    ridgereg = Ridge(alpha=lambda_coeff, normalize=True, solver='sag')
    ridgereg.fit(train_data,target_train)
    y_pred = ridgereg.predict(test_data)
    rmse = sqrt(mean_squared_error(target_test, y_pred))

    return rmse

def preprocess(data):

    res_scaled = preprocessing.scale(data)
    return res_scaled


def formoutput(arr):

    output = pd.DataFrame({'score': arr})
    output.to_csv('output.csv', index=False, header=False)
