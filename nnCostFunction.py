import numpy as np
import sigmoid as s
import sigmoidGradient as sg
def nnCostFunction(nn_params, input_layer_size, hidden_layer1_size,hidden_layer2_size,num_labels, X, y, lambda_reg):
    #Unroll the parameters
    Theta1p= nn_params[hidden_layer1_size * (input_layer_size + 1):]
    Theta2p=Theta1p[(hidden_layer2_size * (hidden_layer1_size+1)):]
    Theta1 = np.reshape(nn_params[:hidden_layer1_size * (input_layer_size + 1)],(hidden_layer1_size, input_layer_size + 1), order='F')
    Theta2 = np.reshape(Theta1p[:hidden_layer2_size * (hidden_layer1_size+1)],(hidden_layer2_size, hidden_layer1_size + 1), order='F')
    Theta3 = np.reshape(Theta2p,(num_labels,hidden_layer2_size + 1), order='F')
    m = len(X)
    J = 0;
    Theta1_grad = np.zeros( Theta1.shape )
    Theta2_grad = np.zeros( Theta2.shape )
    Theta3_grad = np.zeros( Theta3.shape )
    #===========================================================================
    ##Forward propagation using the sigmoid activation function
    mat1=np.ones((m,1))
    X = np.column_stack((mat1, X) ) # = a1
    Z2=np.dot(X,Theta1.T)
    a2 = s.sigmoid(Z2)
    a2 = np.column_stack((mat1, a2) )
    Z3 = np.dot(a2,Theta2.T)
    a3 = s.sigmoid(Z3)
    a3 = np.column_stack(((mat1, a3) ))
    Z4 = np.dot(a3,Theta3.T)
    a4 = s.sigmoid( Z4 )
    y = np.reshape(y,(m,num_labels),order='F')
    #----------------------------------------------------------------------------
    #evaluating the cost function
    cost = np.sum(np.sum((y - a4)**2))
    J = (1.0/(2.0*m))*cost
    sumOfTheta1 = np.sum(np.sum(Theta1[:,1:]**2))
    sumOfTheta2 = np.sum(np.sum(Theta2[:,1:]**2))
    sumOfTheta3 = np.sum(np.sum(Theta3[:,1:]**2))
    J = J + ( (lambda_reg/(2.0*m))*(sumOfTheta1+sumOfTheta2+sumOfTheta3) )
    #Implementing the back propagation
    bigDelta1 = 0
    bigDelta2 = 0
    bigDelta3 = 0
    delta4 = (a4 - y)*sg.sigmoidGradient( Z4 )
    delta3 = (np.dot(delta4,Theta3[:,1:]))*sg.sigmoidGradient(Z3)
    delta2 = np.dot(delta3,Theta2[:,1:]) * sg.sigmoidGradient( Z2 )
    bigDelta1 += np.dot(delta2.T,X)
    bigDelta2 += np.dot(delta3.T,a2)
    bigDelta3 += np.dot(delta4.T,a3)
    Theta1_grad = bigDelta1 / m
    Theta2_grad = bigDelta2 / m
    Theta3_grad = bigDelta3 / m
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta3_grad_unregularized = np.copy(Theta3_grad)
    Theta1_grad += (float(lambda_reg)/m)*Theta1
    Theta2_grad += (float(lambda_reg)/m)*Theta2
    Theta3_grad += (float(lambda_reg)/m)*Theta3
    Theta1_grad[:,0] = Theta1_grad_unregularized[:,0]
    Theta2_grad[:,0] = Theta2_grad_unregularized[:,0]
    Theta3_grad[:,0] = Theta3_grad_unregularized[:,0]
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F'),Theta3_grad.reshape(Theta3_grad.size, order='F')))
    return J, grad
