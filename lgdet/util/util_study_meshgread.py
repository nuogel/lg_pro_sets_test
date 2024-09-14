import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sigma=13.187
rx = np.arange(106)
ry = np.arange(105)
_x = np.arange(212)
_y = np.arange(210)
_xx, _yy = np.meshgrid(_x, _y)

x = np.hstack((rx, rx[::-1]))  # [1...n_1,n-1,...1]
y = np.hstack((ry, ry[::-1]))
xx, yy = np.meshgrid(x, y)
y_reg = np.exp(-1 * (xx ** 2 + yy ** 2) / (sigma ** 2))

fig = plt.figure()
ax = plt.gca(projection='3d')
a = np.arange(y_reg.shape[0])
# # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(_xx, _yy, y_reg, rstride=5, cstride=5, cmap='rainbow')
plt.xlim(0, 220)
plt.ylim(0, 220)

plt.show()