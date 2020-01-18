import numpy as np
import debugInitializeWeights as diw
import nnCostFunction as nncf
import computeNumericalGradient as cng
from decimal import Decimal

def checkNNGradients(lambda_reg=0):
    #CHECKNNGRADIENTS Creates a small neural network to check the
    #backpropagation gradients
    #   CHECKNNGRADIENTS(lambda_reg) Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using computeNumericalGradient). These two gradient computations should
    #   result in very similar values.
    #

    input_layer_size = 2
    hidden_layer1_size = 5
    hidden_layer2_size = 5
    num_labels = 1
    m = 5

    # We generate some 'random' test data
    Theta1 = diw.debugInitializeWeights(hidden_layer1_size, input_layer_size)
    Theta2 = diw.debugInitializeWeights(hidden_layer2_size, hidden_layer1_size)
    Theta3 = diw.debugInitializeWeights(num_labels, hidden_layer2_size)
    # Reusing debugInitializeWeights to generate X
    X  = diw.debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(range(m), num_labels).T

    # Unroll parameters
    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F'), Theta3.reshape(Theta3.size, order='F')))

    # Short hand for cost function
    def costFunc(p):
        return nncf.nnCostFunction(p, input_layer_size, hidden_layer1_size,hidden_layer2_size, \
                   num_labels, X, y, lambda_reg)

    _, grad = costFunc(nn_params)
    numgrad = cng.computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    # code from http://stackoverflow.com/a/27663954/583834
    fmt = '{:<25}{}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

    print('The above two columns you get should be very similar.\n' \
             '(Left Col.: Your Numerical Gradient, Right Col.: Analytical Gradient)')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))

    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). \n' \
             '\nRelative Difference: {:.10E}'.format(diff))
