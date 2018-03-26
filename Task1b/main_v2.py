'''

TASK 1 B - Approach 2 Linear Regression with CV and w/o Regularization

'''
from utils import parsedata
from utils import transform_features
from utils import build_baseline_v2
from utils import formoutput
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import KFold

ids, Y, X = parsedata("train.csv")
X = transform_features(X)


count = 0
min_count = 0
min_rmse = 100000000000
min_w_hat = np.zeros(21)

kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    w_hat, rmse = build_baseline_v2(X_train, X_test, Y_train, Y_test)
    if rmse < min_rmse:
    	min_rmse = rmse
    	min_w_hat = w_hat
    	min_count = count
    count +=1


print(count)
print ("Smaller error found in loop: ", min_count)
print ("Error: ", min_rmse)
print ("W: ", min_w_hat)

Y_pred = np.dot(X, w_hat.T)
full_rmse = (mean_squared_error(Y, Y_pred))**0.5
print ("full rmse: ", full_rmse) 

formoutput(min_w_hat)


