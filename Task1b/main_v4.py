'''

TASK 1 B - Approach 3 Linear Regression with CV and with Regression using Ridge

'''
from utils import parsedata
from utils import transform_features
from utils import build_baseline_v4
from utils import formoutput
import numpy as np
from sklearn.model_selection import KFold

ids, Y, X = parsedata("train.csv")
X = transform_features(X)


lambda_ridge = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
scores = np.zeros((12,len(lambda_ridge)))
min_score = 10000000000000
min_count = 0
count = 0

kf = KFold(n_splits=12, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print ("Train:", train_index, "Test:", test_index)
    for l in range(len(lambda_ridge)):
    	w_hat, rmse = build_baseline_v4(X_train, X_test, Y_train, Y_test, lambda_ridge[l])
    	scores[count,l] = rmse
    count+=1


avg_scores = np.mean(scores, axis = 1)
min_lambda = np.argmin(avg_scores)

min_w_hat, min_rmse = build_baseline_v4(X, X, Y, Y, lambda_ridge[min_lambda])

print(count)
print ("Avg error scores before: ", avg_scores)
print ("Error: ", min_rmse)
print ("W: ", min_w_hat)
formoutput(min_w_hat)