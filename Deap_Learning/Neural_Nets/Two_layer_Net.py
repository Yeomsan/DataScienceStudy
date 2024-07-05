import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import Activation_functions as af
import Loss_functions as lf
from Numerical_Differential import numerical_diff
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        #Reset weights
        self.params = {}
        self.params["W1"] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = af.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = af.softmax(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)

        return lf.cee(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(x, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_diff(loss_W, self.params["W1"])
        grads["b1"] = numerical_diff(loss_W, self.params["b1"])
        grads["W2"] = numerical_diff(loss_W, self.params["W2"])
        grads["b2"] = numerical_diff(loss_W, self.params["b2"])

        return grads
    
