import pandas as pd
import numpy as np
import re
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn import linear_model

df = pd.DataFrame({'temp': [11.9, 14.2, 15.2, 16.4, 17.2, 18.1, 18.5, 19.4, 22.1, 22.6, 23.4, 25.1],
                   'units': [185, 215, 332, 325, 408, 421, 406, 412, 522, 445, 544, 614]})

#https://keras.io/layers/core/#dense
sgd = SGD(lr=1e-5, decay=1e-6, momentum=0, nesterov=False)

model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='normal'))
#model.add(Activation('tanh'))
#model.add(Activation('linear'))
model.add(Activation('relu'))
#model.add(Dense(1, kernel_initializer='glorot_uniform', activation='linear'))
#model.add(Dense(1, kernel_initializer='normal'))
#model.compile(loss='mse', optimizer='adam')
#model.compile(loss='mean_squared_error', optimizer='sgd')
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(df['temp'].values, df['units'].values, epochs=30, batch_size=1)

pred = model.predict(df['temp'].values)
pred

for layer in model.layers:
    weights = layer.get_weights()
    print(layer)
    print(weights)
#h = Wx + bias

model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='normal', activation='relu', use_bias=False))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(df['temp'].values, df['units'].values, epochs=25, batch_size=1)


from keras.datasets import boston_housing
from sklearn.metrics import mean_squared_error
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

pred = model.predict(x_train)
mean_squared_error(pred, y_train)



x = 1.0*np.array(range(200))
y = 2.0*x

model = Sequential()
model.add(Dense(1, input_dim=1))
sgd = SGD(lr=1e-5, decay=1e-6, momentum=0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(x, y, epochs=1, batch_size=1)

score = model.predict(x[:10], batch_size=10)
for layer in model.layers:
    weights = layer.get_weights()
    print(layer)
    print(weights)
print(score)

reg = linear_model.LinearRegression()
reg.fit(df['temp'].values.reshape(12,1), 
        df['units'].values)
reg.coef_
reg.intercept_

#######
x = 1.0*np.array(range(200))
y = x*5 + np.random.rand(200)

model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='glorot_uniform', use_bias=False))
model.compile(loss='mean_squared_error', optimizer=sgd)
#model.fit(x, y, epochs=1, batch_size=10)
#with larger batch size, need more epochs
model.fit(x, y, epochs=10, batch_size=25)

score = model.predict(x[:10], batch_size=10)
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)
print(score)

#######
x = 1.0*np.array(range(200))
y = x**2 + np.random.rand(200)

model = Sequential()
model.add(Dense(3, input_dim=1, kernel_initializer='glorot_uniform', use_bias=False))
model.add(Dense(1, kernel_initializer='glorot_uniform', activation='linear', use_bias=False))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, epochs=1, batch_size=1)

score = model.predict(x[:10], batch_size=1)
lay=[np.zeros(3), np.zeros(3)]
i=0
for layer in model.layers:
    weights = layer.get_weights()
    print(layer)
    print(weights)
    lay[i] = weights[0]
    i += 1
print(score)

a1 = [-0.91426134, -0.42400381] * 2
a2 = [1.1350162, -0.87621868] * 2
np.inner(a1,a2)
np.inner(lay[0] *4, lay[1].transpose())


#######
x1 = 1.0*np.array(range(200))
y = x1**2 + np.random.rand(200)
x=np.vstack([x1, x1]).T

model = Sequential()
model.add(Dense(2, input_dim=2, kernel_initializer='glorot_uniform', use_bias=False))
model.add(Dense(1, kernel_initializer='glorot_uniform', activation='linear', use_bias=False))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, epochs=1, batch_size=1)

score = model.predict(x[:10], batch_size=1)
lay=[np.zeros(2), np.zeros(2)]
i=0
for layer in model.layers:
    weights = layer.get_weights()
    print(layer)
    print(weights)
    lay[i] = weights[0]
    i += 1
print(score)

a1 = [-0.91426134, -0.42400381] * 2
a2 = [1.1350162, -0.87621868] * 2
np.inner(a1,a2)
np.inner(lay[0] *4, lay[1].transpose())

