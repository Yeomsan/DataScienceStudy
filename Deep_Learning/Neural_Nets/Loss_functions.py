import numpy as np

#Sum of Squares for Error (SSE)
def sse(y, t):
    return 0.5 * np.sum((y-t)**2)

#Cross Entropy Error (CEE)
'''
def cee(y, t):
    delta = 1e-7 #to prevent the error of log(0)
    return -np.sum(t * np.log(y + delta))
'''
# but indeed..

def cee(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_s1ze = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_s1ze