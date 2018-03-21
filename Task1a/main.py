from utils import parsedata
from utils import buildmodel
from utils import preprocess
from utils import formoutput
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

ids, Y, X = parsedata("train.csv")

print Y.shape
print X.shape
X_prep = preprocess(X)


lambda_ridge = [0.1, 1, 10, 100, 1000]
scores = np.zeros((10,len(lambda_ridge)))
count = 0

kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X_prep):
    X_train, X_test = X_prep[train_index], X_prep[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    for l in range(len(lambda_ridge)):
        scores[count,l] = buildmodel(X_train,X_test,Y_train,Y_test,lambda_ridge[l])
    count+= 1



formoutput(np.mean(scores, axis=0))


###################################
#scatter_matrix(X.ix[:,2:11])
#plt.show()
#Correl = X.ix[:,2:11].corr()
##################################
