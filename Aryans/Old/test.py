import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Generate some example 3D data
np.random.seed(42)
num_points = 100
x = np.random.rand(num_points)
y = np.random.rand(num_points)
z = np.random.rand(num_points)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c='b', marker='o', label='Random Points')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add a legend
ax.legend()

# Update and redraw the plot every second
num_updates = 50  # Number of updates
for i in range(num_updates):
    new_point = np.random.rand(3)  # Generate new random point
    x = np.append(x, new_point[0])
    y = np.append(y, new_point[1])
    z = np.append(z, new_point[2])
    
    scatter._offsets3d = (x, y, z)  # Update scatter data
    
    plt.pause(1)  # Pause for 1 second

# Show the final plot
plt.show()