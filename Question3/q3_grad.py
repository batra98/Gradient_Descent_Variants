import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def z_function(X,Y):
	Z = (50/9)*((X**2+Y**2)**3) - (209/18)*(X**2+Y**2)**2 + (59/9)*(X**2+Y**2)

	return Z

def get_gradient(w):
	x = w[0]
	y = w[1]

	dell_x = (100/3)*x*((x**2+y**2)**2) - (209/3)*x*(x**2+y**2) + (118/9)*x
	dell_y = (100/3)*y*((x**2+y**2)**2) - (209/3)*y*(x**2+y**2) + (118/9)*y

	w_1 = np.array((dell_x,dell_y))

	error = (50/9)*((x**2+y**2)**3) - (209/18)*(x**2+y**2)**2 + (59/9)*(x**2+y**2)

	return w_1,error 


x1 = np.linspace(-3,3,250)
x2 = np.linspace(-3,3,250)

X,Y = np.meshgrid(x1,x2)

Z = z_function(X,Y)

# def plot_J(X,Y,Z):
#     fig = plt.figure(figsize = (16,8))
#     ax = plt.axes(projection='3d')
#     ax.plot_wireframe(X,Y,Z,color = 'green')
#     ax.set_xlabel('w1')
#     ax.set_ylabel('w2')
#     ax.set_zlabel('J(w1,w2)')
#     plt.show()
    
# plot_J(X,Y,Z)

w = np.array((np.random.random(),np.random.random()))


alpha = 0.001

old_w = []
errors = []

i = 0
while 1:
	gradient,error = get_gradient(w)

	new_w = w - alpha*gradient

	if i%10 == 0:
		print("Iteration: %d - Error: %.4f" % (i, error))
		old_w.append(np.array((new_w[0],new_w[1])))
		errors.append(error)

	if error < 0.001 or i>10000:
		if error < 0.001:
			print("Completed Gradient Descent")
		old_w.append(np.array((new_w[0],new_w[1])))
		errors.append(error)
		break

	w = new_w
	i = i+1

print(w)

all_ws = np.array(old_w)


fig,ax = plt.subplots(1,1)
cp = ax.contour(X,Y,Z,colors = 'black',linestyles = 'dashed',linewidths = 1)
ax.clabel(cp,inline = 1,fontsize = 10)
cp = ax.contourf(X,Y,Z,)
fig.colorbar(cp)
for i in range(len(old_w)-1):
	ax.annotate('',xy = all_ws[i+1,:],xytext=all_ws[i,:],arrowprops={'arrowstyle': '->','color':'r','lw':1},va='center',ha='center')


plt.show()