import numpy as np
from tpot import TPOTClassifier, TPOTRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

x = 1.0*np.array(range(200))
y = x**2 + np.random.rand(200)

reg = linear_model.LinearRegression()
reg.fit(x.reshape(-1,1), y)
print(reg.coef_)
print(reg.intercept_)

pred = reg.predict(x.reshape(-1,1))
mean_squared_error(pred, y)

tpot = TPOTRegressor()
tpot = TPOTRegressor(n_jobs=12, verbosity=2, max_time_mins=5)
tpot.fit(x.reshape(-1,1), y)
print(tpot.score(x.reshape(-1,1), y))
tpot.scoring_function

tpot.export('tpot_exported_pipeline.py')

#####
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Activation, Dense
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

pred = model.predict(x_train)
mean_squared_error(pred, y_train)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
tpot.export('tpot_boston_pipeline.py')