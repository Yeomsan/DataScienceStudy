import numpy as np

#Sum of Squares for Error (SSE)
def sse(y, t):
    return 0.5 * np.sum((y-t)**2)

#Cross Entropy Error (CEE)
def cee(y, t):
    delta = 1e-7 #to prevent the error of log(0)
    return -np.sum(t * np.log(y + delta))