import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy.core.multiarray import dtype


def parsedata(fileTrain):

    train = pd.read_csv(fileTrain)
    nptrain = train.ix[:,:].values

    return train,nptrain

def buildmodel(data,test,target_train,target_test,lambda_coeff):

    # Fit the model

    ridgereg = Ridge(alpha=lambda_coeff, normalize=True)
    ridgereg.fit(data,target_train)
    y_pred = ridgereg.predict(test)
    rmse = sqrt(mean_squared_error(target_test, y_pred))

    return rmse

def preprocess(data):

    res =  np.delete(data,10,1)
    return res



