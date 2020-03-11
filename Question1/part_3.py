import numpy as np
import matplotlib.pyplot as plt

def z_function(w1,w2,x1,x2,e):
	
	summation = 0

	for i in range(400):
		summation = summation + ((w1-2)*x1[i]+(w2-3)*x2[i]+(-e))**2

	return summation


def get_alpha(x):
	alpha = 2*np.dot(x,x.T)

	return alpha


def get_gradient(w,x,e):
	t = np.array((2,3,e))
	w = w-t
	q = 2*np.dot(x,np.dot(x.T,w))
	p = np.dot(np.dot(w.T,x),np.dot(x.T,w))

	# print(q)

	return q,p



x1 = np.random.uniform(-4,4,400)
x2 = np.random.uniform(-4,4,400)

w1 = np.linspace(-20,20,30)
w2 = np.linspace(-20,20,30)
e = np.random.normal(0,0.01)

X,Y = np.meshgrid(w1,w2)

Z = z_function(X,Y,x1,x2,e)




w = np.array((-30,-30,0))
x = np.array((x1,x2,np.ones(400)))

# print(x.shape)


# iterations = 1
# alpha = 0.0001
alpha = get_alpha(x)
alpha = np.linalg.inv(alpha)

alpha = (alpha)

# print(alpha.shape)

errors = []
old_w = []

for i in range(200):
	gradient,error = get_gradient(w,x,e)


	new_w = w - np.dot(alpha,gradient)

	print(gradient)

	if i%1 == 0:
		print("Iteration: %d - Error: %.4f" % (i, error))
	old_w.append(np.array((new_w[0],new_w[1])))
	errors.append(error)

	if error < 0.001:
		old_w.append(np.array((new_w[0],new_w[1])))
		errors.append(error)
		print("Completed Gradient Descent")
		break

	w = new_w

print(w)

all_ws = np.array(old_w)



fig,ax=plt.subplots(1,1)
cp = ax.contour(X, Y, Z,colors = 'black',linestyles = 'dashed',linewidths = 1)
ax.clabel(cp, inline = 1,fontsize = 10)
cp = ax.contourf(X,Y,Z,)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')

for i in range(len(old_w)-1):
	ax.annotate('',xy = all_ws[i+1,:],xytext=all_ws[i,:],arrowprops={'arrowstyle': '->','color':'r','lw':1},va='center',ha='center')


plt.show()

plt.plot(errors)
plt.show()
