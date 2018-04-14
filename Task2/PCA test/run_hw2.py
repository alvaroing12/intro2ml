#import all necessary functions
from utils import *
from pca import *
from pinv import *
from project_utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults

    ##################
    #Exercise 1:

    #Load Iris data
    ids, Y, X =  parsedata("train.csv", 16)

    #Perform a PCA using SVD
    #1. Normalise data by substracting the mean
    X = zeroMean(X)
    Y = zeroMean(Y)
    #2. Compute PCA by computing eigen values and eigen vectors
    [eigen_values,eigen_vectors] = computePCA_SVD(X)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed_data = transformData(eigen_vectors[:,0:3],X)
    
    #5. How much variance can be explained with each principle component?
    sp.set_printoptions(precision=2)
    var = computeVarianceExplained(eigen_values)
    print "Variance Explained SVD: "
    for i in xrange(var.shape[0]):
        print "PC %d: %.2f"%(i+1,var[i])
    print
    #print "Eigen Vectors SVD:"
    #print eigen_vectors
    #print

    #4. Plot your transformed data and highlight the three different sample classes
    plot3DTransform(transformed_data,Y,"3Dsvd.pdf")
