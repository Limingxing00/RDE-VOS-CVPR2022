from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import softmax

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(0.1, 1, 0.05)
Y = np.arange(0.1, 1, 0.05)
X, Y = np.meshgrid(X, Y)
# Z = X*Y # STM
# soft_Z = softmax(Z, axis=0)

Z =  (X/(1-X))/((X/(1-X))+(Y/(1-Y)))

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.xlabel('pi')  #x轴命名
plt.ylabel('pelse')  #y轴命名
plt.show()