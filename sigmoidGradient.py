
import numpy as np

def sigmoidGradient(z):
  #SIGMOIDGRADIENT returns the gradient of the sigmoid function
  #evaluated at z
  #   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
  #   evaluated at z. This should work regardless if z is a matrix or a
  #   vector. In particular, if z is a vector or matrix, you should return
  #   the gradient for each element.


  g = 1.0 / (1.0 + np.exp(-z))
  g = g*(1-g)

  return g

  # =============================================================
