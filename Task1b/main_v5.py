'''

TASK 1 B - Approach 3 Linear Regression with CV and with Regression using Ridge

'''
from utils import parsedata
from utils import transform_features
from utils import build_baseline_v5
from utils import formoutput
import numpy as np
from sklearn.model_selection import KFold

ids, Y, X = parsedata("train.csv")
X = transform_features(X)

formoutput(X, "x_transfrom.csv")

w_hat, rmse, alpha, l1_ratio, regr = build_baseline_v5(X,Y)


print("alpha:", alpha)
print("l1_Ratio:", l1_ratio)
print ("Error: ", rmse)
print ("W: ", w_hat)

formoutput(w_hat, "output.csv")