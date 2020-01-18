import numpy as np
import sigmoid as s
import sigmoidGradient as sg
def nnFit(m,Theta1,Theta2,Theta3, X):
    mat1=np.ones((m,1))
    X = np.hstack((mat1, X) ) # = a1
    a2 = s.sigmoid( np.dot(X,Theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2) )
    a3 = s.sigmoid( np.dot(a2,Theta2.T))
    a3 = np.column_stack((np.ones((a3.shape[0],1)), a3) )
    a4 = s.sigmoid( np.dot(a3,Theta3.T) )
    return a4
