import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_d(z):
	return sigmoid(z)(1-sigmoid(z))


class network_layer:
	def __init__(self,layer_shape,input = False,output = False):
		self.input = input
		self.output = output
		self.previous_weights = np.zeros(layer_shape)
		self.previous_error = np.zeros(layer_shape)
		self.update_value = np.zeros(layer_shape)
		self.inputs = None
		self.weights = None
		self.derivative = None
		self.activated = None

		if self.input == False:
			self.inputs = np.zeros(layer_shape[0])

		if self.output == False:
			self.weights = np.random.normal(size = layer_shape,scale = 0.5)

		def feed_forward(self):
			if self.input == True:
				self.activated = np.hstack(self.inputs,np.ones((self.inputs.shape[0],1)))
				self.output = np.dot(self.activated,self.weights)
			elif self.output == True:
				self.activated = self.output = sigmoid(self.inputs)
				self.derivative = (sigmoid_d(self.inputs)).T
			else:
				self.activated = np.hstack(sigmoid(self.inputs),np.ones((self.inputs.shape[0],1)))
				self.output = np.dot(self.activated,self.weights)
				self.derivative = (sigmoid_d(self.inputs)).T

			return self.output




class neural_network:
	def __init__(self,shape_network,batch_size):
		self.num_layers = len(shape_network)
		self.batch_size = batch_size
		self.shape_network = shape_network
		self.layers = []

		for i in range(self.num_layers):
			if i == (self.num_layers - 1):
				self.layers.append(network_layer((shape_network[i],1),output = True))
			elif i == 0:
				self.layers.append(network_layer((shape_network[i]+1,shape_network[i+1]),input = True))
			else:
				self.layers.append(network_layer((shape_network[i]+1,shape_network[i+1])))

		def train_network(self,X,Y,epochs,update,type):
			results = []

			for e in range(epochs):
				indexes = range(data.shape[0])
				random.shuffle(indexes)

				for i in indexes:
					output = 



data = np.matrix(pd.read_excel("Concrete_Data.xls"))
X = data[:,:-1]
Y = data[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

num_hidden = 25

net = neural_network((X_train.shape[1],num_hidden,Y_train.shape[1]),X_train.shape[0])
