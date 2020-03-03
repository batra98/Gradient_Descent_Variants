import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def z_function(w1,w2,x1,x2,e):
	
	summation = 0

	for i in range(400):
		summation = summation + ((w1-2)*x1[i]+(w2-3)*x2[i]+(1-e))**2

	return summation


x1 = np.random.uniform(-4,4,400)
x2 = np.random.uniform(-4,4,400)

w1 = np.linspace(-10,10,30)
# x1 = x1 - 2
w2 = np.linspace(-10,10,30)
# x2 = x2 - 3
e = np.random.normal(0,0.01)
# x3 = x3-e

X,Y = np.meshgrid(w1,w2)

Z = z_function(X,Y,x1,x2,e)

fig = plt.figure()
ax = plt.axes(projection='3d')


ax.plot_wireframe(X,Y,Z,color = 'green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()