'''
Curiosity : Does an appropriate binary-classifying curve appear if we label 0 or 1 to random dots?

100 dots in (-100, -100), (100, 100) square

By two layer net
'''
import numpy as np
import matplotlib.pyplot as plt

dots = 200 * np.random.rand(100, 2) - 100           #random points
labeled_dots = {}

for (a, b) in dots:
    labeled_dots[(a, b)] = np.random.randint(2)

plt.scatter(dots[0:100, 0], dots[0:100, 1])
plt.show()