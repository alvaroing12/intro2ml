'''

TASK 1 B - Approach 1 Linear Regression w/o CV and w/o Regularization

'''


from utils import parsedata
from utils import transform_features
from utils import build_baseline
from utils import formoutput
import numpy as np
from sklearn.model_selection import KFold

ids, Y, X = parsedata("train.csv")
X = transform_features(X)

w_hat, rmse = build_baseline(X, Y)

print ("Error: ", rmse)
print ("W: ", w_hat)
print (w_hat.shape)

formoutput(w_hat)
