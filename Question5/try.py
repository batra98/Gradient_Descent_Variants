import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Network_Backprop:
	def __init__(self,num_hidden,activation,Jacobian):
		self.num_hidden = num_hidden
		self.num_input = 8
		self.num_output = 1
		self.activation = activation
		self.Jacobian = Jacobian

		self.W1 = np.random.rand(self.num_input,self.num_hidden)
		self.W2 = np.random.rand(self.num_hidden,self.num_output)


	def forward_propagate(self,X,Y):
		X1 = np.dot(X,self.W1)
		Z1 = self.activation(X1)

		X2 = np.dot(Z1,self.W2)


		return X2

	def backpropagate(self,X,Y_,Y):
		# A = self.activation(np.dot(X,self.W1))
		# Jb = self.Jacobian(np.dot(A,self.W2))

		# d_W2 = 2*np.dot(A.T,np.dot(Jb,self.activation(np.dot(A,self.W2))))-2*(np.dot(A.T,np.dot(Jb,Y)))
		# d_W2 = 2*np.dot(A.T,np.dot(A,self.W2))-2*(np.dot(A.T,Y))

		

		

	def train(self,X,Y):
		Y_ = self.forward_propagate(X,Y)
		self.backpropagate(X,Y_,Y)

		

def Jacobian_tanh(z):
	t = 1 - mtanh(z)**2
	t = t.T.tolist()

	return np.diag(t[0])

def mtanh(z):
	return np.tanh(z)




data = np.matrix(pd.read_excel("Concrete_Data.xls"))
X = data[:,:-1]
Y = data[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

net = Network_Backprop(25,mtanh,Jacobian_tanh)

net.train(X_train,Y_train)




