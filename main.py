## Copyright (c) 2019 Ayesh Gunawardana <ayesh@wayne.edu>
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.
################################################################################
### The following code will implemnt a 2 hidden layer feed forward neural network
#Import the follownig phython packages
import scipy.io
import scipy.io as sio
import numpy as np
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#Import the following .py files
# randInitializeWeights.py will initialize the weights randomly
import randInitializeWeights as riw
# nnCostFunction claculate the cost (error) function for each iteration
import nnCostFunction as nncf
# Calculate the gradient of the sigmoid function
import sigmoidGradient as sg
# Calculate the analytical gradient of a sample NN and compare the gradient with backpropagation
import checkNNGradients as cnng
# Fit the NN output
import nnFit as nf
# nnOutput calculate the average and the standard error of the NN fit
import nnOutput as nno
# Load the training set and target set using dataMonteCarlo
import dataMonteCarlo as dmc

###############################################################################3

## Setup the parameters you will use for the two hidden layer neural network.
input_layer_size  = 2;  # numbber of the input layer features: In our code we use q^2 and q^2_{normalized}
hidden_layer1_size = 30;# number of nodes in hidden_layer1
hidden_layer2_size=30;# number of nodes in hidden_layer2
num_labels = 1;          # output of the NN: O(q^2)
# Load Training Data using the dataMonteCarlo.py
print('Loading Data ...')
    # save the data to a .mat file for record keeping
(y1,y2,y3)=dmc.dataMonteCarlo()
a={}
a['x_train']=y1
a['y_train']=y2
a['y_m']=y3
sio.savemat('monteData.mat',a)

#using the scipy read the monteData.mat data file

X=y1
y=y2
y_m=y3
(m,l,q)=X.shape
#Defining the output parameters
TTheta1=np.zeros((hidden_layer1_size,input_layer_size+1,q));
TTheta2=np.zeros((hidden_layer2_size,hidden_layer1_size+1,q));
TTheta3=np.zeros((num_labels,hidden_layer2_size+1,q));
#Implementing Neural networks
#loop through q number of nural networks.
################################################################################
for f in range(q):

    Xn=X[:,:,f]
    yn=y[:,f]
    yn=yn.flatten()
    #random initioalization of the weights
    initial_Theta1 = riw.randInitializeWeights(input_layer_size, hidden_layer1_size);
    initial_Theta2 = riw.randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
    initial_Theta3 = riw.randInitializeWeights(hidden_layer2_size, num_labels);
    #Parameters roll into a column vector initial_nn_params
    initial_nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'), initial_Theta2.reshape(initial_Theta2.size, order='F'),initial_Theta3.reshape(initial_Theta3.size, order='F')))
    #This code is compatible with the regularization as well.
    #After figure out a correct regularization parameter use it below.
    #lambda_reg = 0 is the default value
    lambda_reg = 0
    nn_params=initial_nn_params
    print('Training Neural Network...')
    #Use the scipy optimize.minimize open source minimization algorithm
    #This can be found at https://github.com/scipy/scipy/blob/master/scipy/optimize/lbfgsb.py
    #Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.
    maxiter = 50
    myargs = (input_layer_size, hidden_layer1_size,hidden_layer2_size, num_labels, Xn, yn, lambda_reg)
    results = minimize(nncf.nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)
    nn_params = results["x"]
    #Unroll the parameters from the minimization output
    Theta1p= nn_params[hidden_layer1_size * (input_layer_size + 1):]
    Theta2p=Theta1p[(hidden_layer2_size * (hidden_layer1_size+1)):]
    Theta1 = np.reshape(nn_params[:hidden_layer1_size * (input_layer_size + 1)],(hidden_layer1_size, input_layer_size + 1), order='F')
    Theta2 = np.reshape(Theta1p[:hidden_layer2_size * (hidden_layer1_size+1)],(hidden_layer2_size, hidden_layer1_size + 1), order='F')
    Theta3 = np.reshape(Theta2p,(num_labels,hidden_layer2_size + 1), order='F')
    a4=nf.nnFit(m,Theta1,Theta2,Theta3,Xn)
    TTheta1[:,:,f]= Theta1[:,:];
    TTheta2[:,:,f]= Theta2[:,:];
    TTheta3[:,:,f]= Theta3[:,:];
################################################################################
n1=TTheta1
n2=TTheta2
n3=TTheta3
n4=m
cc={}
cc['TTheta1']=n1
cc['TTheta2']=n2
cc['TTheta3']=n3
cc['m']=n4
#save the NN output to a .mat file name :runData.mat
sio.savemat('run_Data.mat',cc)
#Get the NN average and standard deviation of the NN
(x1,out_put)=nno.nnOutput(X,y,n1,n2,n3,y_m)
# Plot the NN
plt.figure()
plt.errorbar(x1, out_put[:,0], out_put[:,1])
plt.show()
y1=out_put[:,0]
y2=out_put[:,1]
b={}
b['out_put_central_val']=y1
b['out_put_err']=y2
sio.savemat('NNoutput',b)
np.savetxt('nnOutput', (y1,y2))   # x,y,z equal sized 1D arrays
np.savetxt('qSqrVal', (x1))   # x,y,z equal sized 1D arrays
