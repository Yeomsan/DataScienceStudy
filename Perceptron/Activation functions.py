"""
In 'Gates.py' before, we classify the outputs with if-statement.
But we can use not only the 'Heaviside function' but only other nonlinear functions.
We call this 'Activation Function'.
"""
import numpy as np
import matplotlib.pylab as plt

#First Heaviside
def h(x):       #h is an intial of Heaviside function
    y = x >= 0
    return y.astype(np.int)

#Sigmoid Function is often used in neural network
def s(x):
    y = 1 / (1 + np.exp(-x))
    return y

#ReLU ftn is considerable function for these days
def r(x):
    y = np.maximum(0, x)
    return y

#Now you need only calculation with weights and biases.