import sys
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import numpy as np
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from sklearn import linear_model

import tensorflow as tf

print(sys.version)
print(K.backend())

sgd = SGD(lr=1e-5, decay=1e-6, momentum=0, nesterov=False)

x = 1.0 * np.array(range(200))
y = 2.0 * x

model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(x, y, epochs=1, batch_size=1)

score = model.predict(x[:10], batch_size=10)
for layer in model.layers:
    weights = layer.get_weights()
    print(layer)
    print(weights)
print(score)

x = np.random.randn(32, 5)
x2 = np.random.randn(32, 2)

y1 = np.sum(x[:,0:2], 1)
y2 = np.sum(x[:,2:6], 1)

y = np.stack((y1,y2), axis=1)

model = Sequential()
model.add(Dense(100, input_dim=5, batch_size=16, kernel_initializer='glorot_uniform', use_bias=True))
#model.add(Activation('relu'))
model.add(Activation('linear'))
model.add(Dense(2, init='glorot_normal', activation='linear'))
model.compile(loss='mean_squared_error', optimizer=sgd)
model.summary()
model.fit(x, y, epochs=1, batch_size=16)

[x[:,0:1] x[:,1:6]]

model.predict(x, batch_size=16)

model.trainable_weights
critic_grad = tf.placeholder(tf.float32, [None, 1]) 
tf.gradients(model.outputs, model.trainable_weights, -critic_grad)


import keras
x1 = np.random.randn(32, 5)
x2 = np.random.randn(32, 2)
y1 = np.sum(x[:,0:2], 1)
y2 = np.sum(x[:,2:6], 1)
y = np.stack((y1,y2), axis=1)
model1 = Sequential()
model1.add(Dense(100, input_dim=5, batch_size=16, kernel_initializer='glorot_uniform', use_bias=True))
model1.add(Activation('linear'))
model1.add(Dense(2, init='glorot_normal', activation='linear'))

model2 = Sequential()
model2.add(Dense(100, input_dim=2, batch_size=16, kernel_initializer='glorot_uniform', use_bias=True))
model2.add(Activation('linear'))
model2.add(Dense(2, init='glorot_normal', activation='linear'))

iL = [keras.layers.Input(shape=(5,)), keras.layers.Input(shape=(2,))]
merged = keras.layers.Add()([model1(iL[0]), model2(iL[1])])
merged_h1 = Dense(100, activation='relu')(merged)
oL = Dense(2, activation='relu')(merged_h1)
model = keras.models.Model(inputs=iL, outputs=oL)
model.summary()

model.compile(loss='mean_squared_error', optimizer=sgd)
model.summary()
model.fit([x1, x2], y, epochs=1, batch_size=16)
model.predict([x1, x2],batch_size=16)


model1 = keras.models.Sequential()
model1.add(keras.layers.Dense(100, input_shape=(100,)))

model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(100, input_shape=(100,)))

iL = [keras.layers.Input(shape=(100,)), keras.layers.Input(shape=(100,))]
hL = [model1(iL[0]), model2(iL[1])]
oL = keras.layers.Multiply()(hL)

model3 = keras.models.Model(inputs=iL, outputs=oL)


state_input = Input(shape=(3,))
h1 = Dense(24, activation='relu')(state_input)
h2 = Dense(48, activation='relu')(h1)
h3 = Dense(24, activation='relu')(h2)
output = Dense(1, activation='relu')(h3)
        
model = Model(input=state_input, output=output)
adam  = Adam(lr=0.001)
model.compile(loss="mse", optimizer=adam)
model.summary()

model.outputs
model.trainable_weights
model_weights = model.trainable_weights
critic_grad = tf.placeholder(tf.float32, [None, 1]) 
grads = tf.gradients(model.outputs, model_weights, -critic_grad)
grads = zip(grads, model_weights)
optimize = tf.train.AdamOptimizer(.001).apply_gradients(grads)

state_input = Input(shape=(3,))
state_h1 = Dense(24, activation='relu')(state_input)
state_h2 = Dense(48)(state_h1)
action_input = Input(shape=(1,))
action_h1    = Dense(48)(action_input)     
merged    = Add()([state_h2, action_h1])
merged_h1 = Dense(24, activation='relu')(merged)
output = Dense(1, activation='relu')(merged_h1)
model  = Model(input=[state_input,action_input], output=output)
model.compile(loss="mse", optimizer=adam)
model.summary()

#planets x rounds x data dim
x = np.random.randn(5, 20, 15)
x.shape
#output is # rounds
y = np.random.randn(5, 20, 1)
#y = y.reshape((1, 5, 20))
y.shape

# expected input data shape: (batch_size, timesteps, data_dim)
#(samples, timesteps, dimension)
model = Sequential()
model.add(Dense(100, input_shape=(20, 15), kernel_initializer='glorot_uniform'))
model.add(Activation('linear'))
model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
model.compile(loss='mean_squared_error', optimizer=adam)
model.summary()
model.fit(x, y, epochs=1, batch_size=5)
#want pred for every round
model.predict(x, batch_size=5).shape


model = Sequential()
model.add(Dense(32, input_shape=(20, 15)))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(2))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)
model.summary()
y = model.predict(x, batch_size=5)
y.shape

x = np.random.randn(5, 20, 15)
x.shape
x = x.reshape((20, 5*15))

model = Sequential()
model.add(Dense(32, input_dim=5*15))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(2))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)
model.summary()
y = model.predict(x)
y.shape



x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
x.shape
x = x.reshape((1, 5, 3))
x.shape

model = Sequential()
model.add(Dense(32, input_shape=(5, 3)))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)
model.summary()

y = model.predict(x)
y.shape

x = np.stack((x, x1))
x.shape