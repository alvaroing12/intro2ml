import scipy as sp
import scipy.linalg as linalg
#import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import plot_color

'''############################'''
'''Principle Component Analyses'''
'''############################'''

'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''
def computeCov(X=None):
    Xm = X - X.mean(axis=0)
    return 1.0/(Xm.shape[0]-1)*sp.dot(Xm.T,Xm)

'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
    #compute eigen values and vectors
    [eigen_values,eigen_vectors] = linalg.eig(matrix)
    #sort eigen vectors in decreasing order based on eigen values
    indices = sp.argsort(-eigen_values)
    return [sp.real(eigen_values[indices]), eigen_vectors[:,indices]]

'''
Compute PCA using SVD
Input: Data Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use SciPy svd solver!
'''
def computePCA_SVD(matrix=None):
    #Normalise input matrix to be unbiased
    X = 1.0/sp.sqrt(matrix.shape[0]-1) * matrix
    #Compute SVD in X
    [L,S,R] = linalg.svd(X)
    #Variance is singular values squared
    eigen_values = S*S
    eigen_vectors = R.T
    return [eigen_values,eigen_vectors]

'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs=None,data=None):
    return sp.dot(data,pcs)

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):
    return evals/evals.sum()


'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    #Save file
    pl.savefig(filename)

'''
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename=None):
    pl.figure()
    ind_l = sp.unique(labels)
    legend = []
    for i,label in enumerate(ind_l):
        ind = sp.where(label==labels)[0]
        plot = pl.scatter(transformed[ind,0],transformed[ind,1],color=plot_color[i],alpha=0.5)
        legend.append(plot)
    pl.legend(ind_l,scatterpoints=1,numpoints=1,prop={'size':8},ncol=6,loc="upper right",fancybox=True)
    pl.xlabel("Transformed X Values")
    pl.ylabel("Transformed Y Values")
    pl.grid(True)
    #Save File
    if filename!=None:
       pl.savefig(filename)

def plot3DTransform(transformed=None, labels=None, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ind_l = sp.unique(labels)
    
    for i,label in enumerate(ind_l):
        ind = sp.where(label==labels)[0]
        ax.scatter(xs=transformed[ind,0], ys=transformed[ind,1], zs=transformed[ind,2], zdir='z', s=20, c=plot_color[i], marker='o')    
    
    ax.set_xlabel('Transformed X Values')
    ax.set_ylabel('Transformed Y Values')
    ax.set_zlabel('Transformed Z Values')
    plt.show()

'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    Xm = X - X.mean(axis=0)
    return Xm/sp.std(Xm,axis=0)

'''
Substract Mean from Data (zero mean)
'''
def zeroMean(X=None):
    return X - X.mean(axis=0)
