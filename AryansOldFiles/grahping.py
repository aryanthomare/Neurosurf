import matplotlib.pyplot as plt
import math
import perlin
vals = []
ax = []
p = perlin.Perlin(999) #6789 is the seed

for y in range(0,10000):
    x = y/10
    vals.append(p.one(x))
    ax.append(x)

plt.plot(ax, vals, 'r')
plt.show()