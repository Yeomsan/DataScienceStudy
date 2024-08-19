'''
Curiosity : Does an appropriate binary-classifying curve appear if we label 0 or 1 to random dots?

100 dots in (-100, -100), (100, 100) square

By two layer net
'''
import numpy as np
import matplotlib.pyplot as plt

dots_num = 100

x = np.random.uniform(-100, 100, dots_num)
y = np.random.uniform(-100, 100, dots_num)

labels = np.random.randint(0, 2, num_points)

plt.figure(figsize=(8, 8))
plt.scatter(x[labels == 0], y[labels == 0], color='red', label='Label 0')
plt.scatter(x[labels == 1], y[labels == 1], color='blue', label='Label 1')

# Set plot limits
plt.xlim(-120, 120)
plt.ylim(-120, 120)

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Random Points with Labels')
plt.legend()

# Show plot
plt.grid(True)
plt.show()