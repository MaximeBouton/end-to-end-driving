

############################################################################
#      REGRESSION USING CNN
#
#
############################################################################

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np


############################################################################
# PARSE COMMAND LINE ARGUMENT
############################################################################

import argparse

parser = argparse.ArgumentParser(description='CNN Parameters')

# filter size 
parser.add_argument('--filter', default=3)
# epochs 
parser.add_argument('--epochs', default=30)
# reduction factor 
parser.add_argument('--factor', default=8.0)

# parse args 
args = parser.parse_args()


############################################################################
# CNN AND DATASET  PARAMETERS 
############################################################################

batch_size = 32
nb_epoch = int(args.epochs)
data_augmentation = False
fs = int(args.filter)

# input image dimensions
factor  = float(args.factor)
img_rows, img_cols = int(480/factor), int(640/factor)
# the images are RGB
img_channels = 3



# the data, shuffled and split between train val and test sets
dataset_name = '../../data/resized_features' + str(int(factor)) + '.npy'
X = np.load(dataset_name)
y = np.load('regLabels.npy')

# zero mean labels
mean = y.mean()
std  = y.std()
y = (y - mean) / std


# shuffle the data
np.random.seed(3)
p = np.random.permutation(len(y))
y = y[p]
X = X[p,:,:,:]

fr_tr = .8
fr_va = .1
## Repartition 

(X_train, y_train) = X[:int(fr_tr*X.shape[0])], y[:int(fr_tr*X.shape[0])]
(X_val,  y_val)    = X[int(fr_tr*X.shape[0]):int((fr_tr+fr_va)*X.shape[0])], y[int(fr_tr*X.shape[0]):int((fr_tr+fr_va)*X.shape[0])]
(X_test, y_test)   = X[int((fr_tr+fr_va)*X.shape[0]):], y[int((fr_tr+fr_va)*X.shape[0]):]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')

############################################################################
#     MODEL DEFINITION                                                     #
############################################################################

#act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

model = Sequential()

model.add(Convolution2D(32, fs, fs, border_mode='same',
                        input_shape=(img_rows, img_cols,3)))
#model.add(Activation('relu'))
model.add(PReLU(init='zero', weights=None))
model.add(Convolution2D(32, fs, fs))
#model.add(Activation('relu'))
model.add(PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(64, fs, fs))
#model.add(Activation('relu'))
model.add(PReLU(init='zero', weights=None))
model.add(Convolution2D(64, fs, fs))
#model.add(Activation('relu'))
model.add(PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
#model.add(Activation('relu'))
model.add(PReLU(init='zero', weights=None))
model.add(Dense(32))
#model.add(Activation('relu'))
model.add(PReLU(init='zero', weights=None))
#model.add(Dropout(0.5))
model.add(Dense(1))
#model.add(Activation('softmax'))

# Define a Callback function to save the weight and measure test error

checkpointer = ModelCheckpoint(filepath='tmp/weights.{epoch:02d}', verbose=1, save_best_only=False)

#tfboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)


############################################################################
#     Training                                                             #
############################################################################

# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer='adam')

X_train = X_train.astype('float32')
X_val   = X_val.astype('float32')
X_test  = X_test.astype('float32')

X_train /= 255
X_val   /= 255
X_test  /= 255

print('Not using data augmentation.')
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    validation_data=(X_val, y_val),
                    shuffle=True,
                    callbacks = [checkpointer])

# estimator = KerasRegressor(model, nb_epoch=nb_epoch)


############################################################################
#     Test and plot loss function                                          #
############################################################################

testLoss = model.evaluate(X_test,y_test,
	       batch_size = batch_size)

print('Testing Loss = {}'.format(testLoss))


#### save training stats

import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('loss.png')
# plt.show()

np.save('train_loss.npy',history.history['loss'])
np.save('val_loss.npy',history.history['val_loss'])
