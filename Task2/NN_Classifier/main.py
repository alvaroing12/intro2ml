'''

TASK 2 - Neural Network Classifier Approach

'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import parsedata, formoutput
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt




'''
### 00. Test class imbalance ###
unique, counts = np.unique(y_train, return_counts=True)
print dict(zip(unique, counts))

#=> results show almost identical separation of 1/3 for each class


### 0. Plot histograms of every feature ###
for i in range(X_train.shape[1]):
	# the histogram of the data
	n, bins, patches = plt.hist(X_train[:,i], 50, normed=1, facecolor='green')

	# add a 'best fit' line
	plt.xlabel('x{0}'.format(i+1))
	plt.ylabel('Distribution')
	plt.title('Histogram of feature x{0}'.format(i+1))
	plt.grid(True)
	plt.savefig('hist_train_x{0}'.format(i+1))

for i in range(X_test.shape[1]):
	# the histogram of the data
	n, bins, patches = plt.hist(X_test[:,i], 50, normed=1,facecolor='green')

	# add a 'best fit' line
	plt.xlabel('x{0}'.format(i+1))
	plt.ylabel('Distribution')
	plt.title('Histogram of feature x{0}'.format(i+1))
	plt.grid(True)
	plt.savefig('hist_test_x{0}'.format(i+1))
'''


### 1. Split Tr + Va, Te sets from training data ###
ids, y, X = parsedata("train.csv", 16)
#X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_cv = X
y_cv = y
print("X_cv:",X_cv.shape)
#print("X_test:",X_test.shape)
print("y_cv:",y_cv.shape)
unique, counts = np.unique(y_cv, return_counts=True)
#print("y_test:",y_test.shape)
#unique, counts = np.unique(y_cv, return_counts=True)
### 2. Train Network on Tr dataset ####

alphas = [1]
nn_clfs = []
train_values = []
valid_values = []
train_complete_values = []
valid_complete_values = []
for i in alphas:
	nn_clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1, max_iter=625, warm_start=True, hidden_layer_sizes=(8,), random_state=45236)
	nn_clfs.append(nn_clf)


skfold = StratifiedKFold(n_splits=10, random_state=36851234)
for nn_clf in nn_clfs:
	aux = []
	for train_indices, test_indices in skfold.split(X_cv,y_cv):
		nn_clf.fit(X_cv[train_indices], y_cv[train_indices])
		aux.append(nn_clf.score(X_cv[test_indices], y_cv[test_indices]))
	train_values.append(np.mean(aux))
	#y_pred = nn_clf.predict(X_test)
	#valid_values.append(accuracy_score(y_test, y_pred))

print(train_values)
'''
print(valid_values)
print
print(train_complete_values)
print(valid_complete_values)

plt.plot(range(len(alphas)),train_values, c='b')
plt.plot(range(len(alphas)),valid_values, c='r')
plt.show()
'''


### 3. Improve hyperparameters on Va dataset ####
#acc_Va = accuracy_score(y, y_pred)

### 4. See accuracy vs. precision vs. recall score on Te dataset ###
#acc_Te = accuracy_score(y, y_pred)

### 5. Predict y label for real Test data ###
ids_test, _ , X_test = parsedata("test.csv", 16, y_label_present=False)
y_pred = nn_clf.predict(X_test)
formoutput(ids_test, y_pred, "output.csv",use_header=True)

