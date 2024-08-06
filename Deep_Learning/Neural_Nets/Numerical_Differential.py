import numpy as np
"""
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return ((f(x+h)-f(x-h)) / (2*h))
    """

#indeed..

def numerical_diff(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def ftn(x):
    return x[0]**2 + x[1]**2

print(numerical_diff(ftn, np.array([3.0, 4.0])))
