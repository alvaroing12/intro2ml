'''

TASK 1 B - Approach 1 Linear Regression w/o CV and w/o Regression

'''


from utils import parsedata
from utils import transform_features
from utils import build_baseline
from utils import formoutput
import numpy as np
from sklearn.model_selection import KFold

ids, Y, X = parsedata("train.csv")

X_transform = transform_features(X)

w_hat, rmse = build_baseline(X_transform, Y)

print ("Error: ", rmse)
print ("W: ", w_hat)


formoutput(w_hat)
