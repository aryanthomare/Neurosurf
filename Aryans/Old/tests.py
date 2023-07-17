import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Generate random data
x = np.linspace(0, 10, 100)
y1 = np.random.rand(100)
y2 = np.random.rand(100)
y3 = np.random.rand(100)

# Create the figure and define the grid layout
fig = plt.figure(figsize=(10, 6))
grid = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])

# Create the top graph spanning the whole width
ax1 = fig.add_subplot(grid[0, :])
ax1.plot(x, y1)
ax1.set_title("Top Graph")
ax1.set_xlabel("X Label")
ax1.set_ylabel("Y Label")

# Create the bottom-left graph
ax2 = fig.add_subplot(grid[1, 0])
ax2.plot(x, y2)
ax2.set_title("Bottom-Left Graph")
ax2.set_xlabel("X Label")
ax2.set_ylabel("Y Label")

# Create the bottom-right graph
ax3 = fig.add_subplot(grid[1, 1])
ax3.plot(x, y3)
ax3.set_title("Bottom-Right Graph")
ax3.set_xlabel("X Label")
ax3.set_ylabel("Y Label")

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.3)

# Show the figure
plt.show()