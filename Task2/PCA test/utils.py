import scipy as sp
import scipy.linalg as linalg
import numpy as np
from sklearn import datasets
import pylab as pl
import matplotlib as mpl


'''
Gobal Color For Plotting
'''
plot_color = ['#F7977A','#FDC68A','#A2D39C','#6ECFF6','#8493CA','#BC8DBF','#F6989D','#FFF79A']

def initPlotLib():
    pl.ion()
    font_size = 10
    mpl.rcParams['font.family']="sans-serif"
    mpl.rcParams['font.sans-serif']="Arial"
    mpl.rcParams['font.size']=font_size
    mpl.rcParams['font.weight']='medium'
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['patch.edgecolor'] = '#000000'
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['grid.color'] = '#AAAAAA'

'''
General Data Container for Simulated Data
'''
class Data(dict):
    def __init__(self,**kwargs):
        dict.__init__(self,kwargs)

    def __setattr__(self,key=None,value=None):
        self[key] = value

    def __getattr__(self,key=None):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__

'''
Function to generate simulation data for PCA
'''
def generate_simulation_data(n_classes=2,n_samples=[20,20],n_features=3,seed=0,scales=None):
    #initial checks
    assert n_classes>0, "n_classes has to be larger than 0"
    assert n_features>2, "n_features has to be larger than 2"
    assert n_classes==len(n_samples), "n_samples has to be an array with as many elements as n_classes"
    if sp.any(scales)==None:
        scales = sp.ones(n_features)
    else:
        assert len(scales)==n_features, "scales has to be an array with as many elements as features"

    #set seed
    sp.random.seed(seed)
    
    feature_matrix = None
    class_labels = None
    #generate data for each class
    for i in xrange(n_classes):
        #generate random data for class i drawn from a multivariate gauss distribution
        class_matrix = sp.random.multivariate_normal(sp.ones(n_features)*(i+1)*sp.random.randn(n_features),sp.eye(n_features),n_samples[i])
        #store data
        class_labels = sp.ones(n_samples[i])*(i+1) if (sp.any(class_labels)==None) else sp.concatenate([class_labels,sp.ones(n_samples[i])*(i+1)])
        feature_matrix = class_matrix if (sp.any(feature_matrix)==None) else sp.vstack([feature_matrix,class_matrix])
    #scale features
    if sp.any(scales)!=None:
        feature_matrix *= scales
    #generate data dict
    data = Data(data=feature_matrix,
                target=class_labels,
                n_samples=feature_matrix.shape[0],
                n_features=feature_matrix.shape[1])
    return data

'''
Simulate Data with predifined Settings
'''
def simulateData():
    #init parameters
    n_samples = 50 #number of samples per class
    n_features = 50 #total number of features
    n_classes = 5 #total number of classes
    
    #set some elements to different scales
    scale_elements = 10 #total number of elements to rescale
    scales = sp.ones(n_features)
    sp.random.seed(0)
    ind = sp.random.randint(1,n_features,scale_elements)
    scales[ind] = ind
    
    #Generate Random Data drawn from a multivariate gaussian distribution 
    #(each class has a different average mean)
    data = generate_simulation_data(n_classes=n_classes,
                                    n_samples=n_classes*[n_samples],
                                    n_features=n_features,
                                    seed=0,
                                    scales=scales)
    return data

'''
Load Iris Data
'''
def loadIrisData():
    return datasets.load_iris()
