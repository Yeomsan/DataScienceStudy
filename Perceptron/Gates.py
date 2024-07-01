import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = -1
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 0
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-1, -1])
    b = 2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

"""         can use for test
while True:
    x,y=map(int,input().split())
    print(x, y, "AND => ", AND(x, y), "OR => ", OR(x, y), "NAND => ", NAND(x, y))
"""
