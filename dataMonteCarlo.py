#This function creates the file monteData.mat. This creates the training set and
# target to train the neural network. The input data is provided in the .csv form.
#Import python packages
import scipy
import io
import numpy as np
import csv
import pandas as pd
import scipy.io as sio
################################################################################
def dataMonteCarlo():
    listX=[]
    listY=[]
    # Read the data file in .csv form
    # Ex. For the NN tarining we used 2.8 million pseudo data
    with open('DtoPi_PseudoData_1.csv') as data:
        readCSV=csv.reader(data, skipinitialspace=False,delimiter=',', quoting=csv.QUOTE_NONE)
        for row in readCSV:
            listX.append(float(row[0]))
            listY.append(float(row[1]))
    Q_artq=np.array([listX])
    O_art=np.array([listY])
    rr1,cc1=O_art.shape
    # Seperate the data set into sub sets.
    # Ex. The data set is seperated for 100 batches. Each batch has 28000 training data
    # 100 NN will be trained based on each data batch
    Data_points_per_batch=28000
    intr=Data_points_per_batch
    # Seperate the data set into bin number of batches
    bin=int(cc1/intr)
    O_arts=np.zeros((rr1,intr,bin))
    Q_artsq=np.zeros((rr1,intr,bin))
    y_m=np.zeros((1,bin))
    x_train=np.zeros((rr1*intr,2,bin))
    y_train=np.zeros((rr1*intr,bin))
    # loop through the bin number of batches
    for k in range(bin):
        O_arts[:,:,k]=O_art[:,(k)*intr:(k+1)*intr]
        Q_artsq[:,:,k]=Q_artq[:,(k)*intr:(k+1)*intr]
        O_art1=np.reshape(O_arts[:,:,k],rr1*intr,1)
        Q_art1=np.reshape(Q_artsq[:,:,k],rr1*intr,1)
        o_mat= np.column_stack((Q_art1, O_art1))
        a=o_mat
        m,n=a.shape
        idx = np.random.permutation(m)
        b = a
        b[idx,:] = a[:,:]
        x_data=a[:,0]
        y_datau=a[:,1]
        mu = np.mean(x_data, axis=0) #axis=1 runs along the columns horizontally
        X_norm=x_data-mu
        sigma = np.std(X_norm)
        X_norm=X_norm/sigma
        X_norm.shape
        x_data=np.column_stack((x_data, X_norm))
        y_m[:,k]=y_datau.max()
        y_data=y_datau/(y_m[:,k])
        x_train[:,:,k]=x_data[:,:]
        y_train[:,k]=y_data
    # Ex. Training set is 28000*2*100 3d matrix
    # Ex. Target is 28000*1 matrix
    y1=x_train
    y2=y_train
    y3=y_m
    return y1,y2,y3
