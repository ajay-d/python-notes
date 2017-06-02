import sys
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from sklearn import linear_model

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
