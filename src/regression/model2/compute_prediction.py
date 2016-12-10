############################################################
#                                                          #
#        Compute steering prediction on the whole data set #
###########################################################

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np

### Some parameters
batch_size = 32
nb_classes = 3
nb_epoch =  30 
fs = 3
data_augmentation = False

# input image dimensions
factor  = 8.0
img_rows, img_cols = int(480/factor), int(640/factor)
# the images are RGB
img_channels = 3

### Load DATA  
dataset_name = '../../../data/resized_features' + str(int(factor)) + '.npy'
X = np.load(dataset_name)
y = np.load('../regLabels.npy')

### Convert the DATA 

mean = y.mean() 
std = y.std() 
y = (y-mean) / std

X /= 255.

### Generate Model 

model = Sequential()

model.add(Convolution2D(32, fs, fs, border_mode='same',
                        input_shape=(img_rows, img_cols,3)))
model.add(PReLU(init='zero', weights=None))
model.add(Convolution2D(32, fs, fs))
model.add(PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, fs, fs))
model.add(PReLU(init='zero', weights=None))
model.add(Convolution2D(64, fs, fs))
model.add(PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, fs, fs))
model.add(PReLU(init='zero', weights=None))
model.add(Convolution2D(64, fs, fs))
model.add(PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(PReLU(init='zero', weights=None))
model.add(Dense(32))
model.add(PReLU(init='zero', weights=None))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
                     optimizer='adam')
### Load weights and compute 

weights = 'tmp3/weights.29'
model.load_weights(weights)

m = len(y)
predictions = model.predict(X, batch_size=batch_size, verbose=1)

np.save('predictions.npy',predictions)

import matplotlib.pyplot as plt

plt.plot(y,'b')
plt.plot(predictions,'r')
plt.title('Predicted Steering commands and Real commands')
plt.legend(['Real comand','Predicted command'])
plt.show()



