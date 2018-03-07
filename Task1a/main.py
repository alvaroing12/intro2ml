from utils import parsedata
from utils import buildmodel
from utils import preprocess
import numpy as np
from sklearn.model_selection import KFold

X,npX = parsedata("train.csv")

train_npX = npX[:,1:12]   # leave id out

train_npX_pro = preprocess(train_npX)  # preprocess data. Last column is deleted

lambda_ridge = [0.1, 1, 10, 100, 1000]
scores = np.zeros((10,len(lambda_ridge)))
count = 0

kf = KFold(n_splits=10)
for train, test in kf.split(npX):
    for l in range(len(lambda_ridge)):
        scores[count,l] = buildmodel(train_npX_pro[train,1:],train_npX_pro[test,1:],train_npX_pro[train,0],train_npX_pro[test,0],lambda_ridge[l])
    count+= 1


#Correl = train_npX_pro.corr()
#print np.unique(npTrain[:,11])
