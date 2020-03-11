import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

data = np.matrix(pd.read_excel("Concrete_Data.xls"))
X = data[:,:-1]
Y = data[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

hidden_layers = 25
activation = 'tanh'

Y_test = np.ravel(Y_test)
Y_train = np.ravel(Y_train)

regressor = MLPRegressor(
			hidden_layer_sizes = (hidden_layers,),
			activation = activation,
			batch_size = X_train.shape[0],
			shuffle = True,
			max_iter = 1000
			)

regressor.fit(X_train,Y_train)

Y = regressor.predict(X_test)
Y_ = regressor.predict(X_train)

print(regressor.loss_curve_)
print((np.square(Y-Y_test).mean(axis=0)))
print((np.square(Y_-Y_train).mean(axis=0)))
# print(np.dot((Y_-Y_train),Y_ - Y_train))





