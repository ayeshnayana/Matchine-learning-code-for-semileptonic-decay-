import scipy
import io
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import sigmoid as s
import sigmoidGradient as sg
import nnFit as nf
###############################################################################
def nnOutput(X,y,n1,n2,n3,y_m):
    (m,l,q)=X.shape
    n=140000
    X1m=np.amax(X[:,0])
    X2min=np.amin(X[:,1])
    X2max=np.amax(X[:,1])
    x1=np.linspace(0,X1m,n)
    x2=np.linspace(X2min,X2max,n)
    x1t=np.transpose(x1)
    x2t=np.transpose(x2)
    X_data=np.column_stack((x1, x2))
    temp_fit=np.zeros((n,q))
    out_put=np.zeros((n,2))
    for i in range(q):
        Theta1=n1[:,:,i]
        Theta2=n2[:,:,i]
        Theta3=n3[:,:,i]
        a=nf.nnFit(n,Theta1,Theta2,Theta3, X_data)*y_m[0,i]
        temp_fit[:,i]=np.reshape(a,n,1)
    (mm,nn)=temp_fit.shape
    for j in range(mm):
        out_put[j,0]=np.mean(temp_fit[j,:])
        out_put[j,1]=np.std(temp_fit[j,:])
    return x1,out_put
