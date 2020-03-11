import numpy as np
import matplotlib.pyplot as plt

# def my_tanh(z):
# 	return np.tanh(z)


# def Jacobian(A):
# 	# print(my_tanh(A))
# 	t = 1-my_tanh(A)**2
# 	t = np.array(t)

# 	return np.diag(t)

def get_gradient(W,X,Y):
	G = np.dot(X,W)
	J = np.dot(G-Y,G-Y)
	# A = Jacobian(np.dot(X,W))
	# J_g = 2*np.dot(np.dot(X.T,A),my_tanh(np.dot(X,W)))-2*np.dot(np.dot(X.T,A),Y)
	J_g = 2*(np.dot(np.dot(X.T,X),W)-np.dot(X.T,Y))

	return J_g,J




u1 = np.array((-3,4))
u2 = np.array((4,-3))

Cov = np.array(((16,0),(0,9)))

train_x1 = np.random.multivariate_normal(u1,Cov,200)
train_y1 = np.ones(200)
train_x2 = np.random.multivariate_normal(u2,Cov,200)
train_y2 = -np.ones(200)

test_x1 = np.random.multivariate_normal(u1,Cov,100)
test_y1 = np.ones(100)

test_x2 = np.random.multivariate_normal(u2,Cov,100)
test_y2 = -np.ones(100)


m1 = np.mean(np.hstack((train_x1.T,train_x2.T)).T,axis = 0)
m2 = np.std(np.hstack((train_x1.T,train_x2.T)).T,axis = 0)

# l1 = np.mean(train_x2,axis = 0)
# l2 = np.std(train_x2,axis = 0)

for i in range(200):
	train_x1[i] = (train_x1[i]-m1)/(m2)
	train_x2[i] = (train_x2[i]-m1)/(m2)



X = np.hstack((np.vstack((train_x1.T,train_y1)),np.vstack((train_x2.T,train_y1)))).T
Y = np.hstack((train_y1,train_y2))
W = np.array((np.random.normal(),np.random.normal(),1))


X_test = np.hstack((np.vstack((test_x1.T,test_y1)),np.vstack((test_x2.T,test_y1)))).T
Y_test = np.hstack((test_y1,test_y2))
# W = np.array((0,0,1))


# A = np.dot(X,W)

# Jacobian(A)

alpha = 0.00001
errors = []
errors_test = []
old_W = []

for i in range(1000):
	gradient,error = get_gradient(W,X,Y)

	new_W = W - alpha*gradient


	if i%1 == 0:
		print("Iteration: %d - Error: %.4f" % (i, error))
	old_W.append(np.array((new_W[0],new_W[1])))
	errors.append(error)
	errors_test.append(200-np.count_nonzero(np.sign(np.dot(X_test,W)) == Y_test))

	if error < 0.001:
		old_w.append(np.array((new_W[0],new_W[1])))
		errors.append(error)
		print("Completed Gradient Descent")
		break

	W = new_W

print(W)


plt.plot(errors)
plt.show()

A = np.sign(np.dot(X_test,W))
print(A)
print(np.count_nonzero(A==Y_test)*100.0/200)

plt.plot(errors_test)
plt.show()





