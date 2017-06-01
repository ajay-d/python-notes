from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train.shape
X_test.shape

X_train.ndim
X_train.size
X_train.dtype
X_train.dtype.name

#each obs/pixel is defined by 3 colors
X_train[0,0,0]

y_train.shape
y_train[:10]

np.unique(y_train)

num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 10 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

model = Sequential()
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu',
                        input_shape=(height, width, depth)))
model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_1))
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu'))
model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_2))
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy
model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

model.evaluate(X_test, Y_test, verbose=1)

