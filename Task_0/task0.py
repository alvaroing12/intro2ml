import pandas as pd
import numpy as np
from numpy.core.multiarray import dtype
from sklearn.metrics import mean_squared_error

fileTrain = "train.csv"
fileTest = "test.csv"


# Fake training
#read training data from file using pandas
df = pd.read_csv(fileTrain)

Ids = df.ix[:,0]
Ys = df.ix[:,1]
Xs = df.ix[:,2:12]


Ys_pred = Xs.mean(axis = 1, numeric_only=True)
RMSE = mean_squared_error(Ys, Ys_pred)**0.5

print RMSE

# Test step

df = pd.read_csv(fileTest)
Ids = df.ix[:,0]
Xs = df.ix[:,1:11]
Ys_pred = Xs.mean(axis = 1, numeric_only=True)

output = pd.DataFrame({'Id': Ids, 'y': Ys_pred})
output.to_csv('output.csv', index=False)





