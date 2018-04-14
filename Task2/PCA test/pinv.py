import scipy as sp
import scipy.linalg as linalg

'''############################'''
'''Moore Penrose Pseudo Inverse'''
'''############################'''

'''
Compute Moore Penrose Pseudo Inverse
Input: X: matrix to invert
       tol: tolerance cut-off to exclude tiny singular values (default=1e15)
Output: Pseudo-inverse of X.
Note: Do not use scipy or numpy pinv method. Implement the function yourself.
      You can of course add an assert to compare the output of scipy.pinv to your implementation
'''
def compute_pinv(X=None,tol=1e-15):
    #Compute SVD
    [L,d,R] = linalg.svd(X,False)
    #Compute reciprocals
    for i in xrange(d.shape[0]):
        if d[i]>tol:
            d[i] = 1.0/float(d[i])
        else:
            d[i] = 0.0
    return sp.dot(R.T,sp.diag(d)).dot(L.T)
