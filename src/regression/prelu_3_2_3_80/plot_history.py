########### SCRIPT TO VISUALIZE CNN OUTPUT ################

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
import matplotlib.pyplot as plt


### Some parameters
batch_size = 32
nb_epoch = 30
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

mean = y.mean()
std = y.std() 
y = (y-mean)/std


# shuffle the data 
np.random.seed(3)
p = np.random.permutation(len(y))
y = y[p]
X = X[p,:,:,:]

(X_train, y_train) = X[:int(.8*X.shape[0])], y[:int(.8*X.shape[0])]
(X_val,  y_val)    = X[int(.8*X.shape[0]):int(.9*X.shape[0])], y[int(.8*X.shape[0]):int(.9*X.shape[0])]
(X_test, y_test)   = X[int(.9*X.shape[0]):], y[int(.9*X.shape[0]):]

X_train = X_train.astype('float32')
X_val   = X_val.astype('float32')
X_test  = X_test.astype('float32')

X_train /= 255
X_val   /= 255
X_test  /= 255


### Generate Model and evaluate for each weights 

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
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
             optimizer='adam')
### Compute test acc and loss

test_loss = np.zeros(nb_epoch)
for i in range(nb_epoch):
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
    # load pre-trained weights
    weights = 'tmp/weights.%02d'%i 
    model.load_weights(weights)
    model.compile(loss='mean_squared_error', optimizer='adam')
    test_loss[i]  = model.evaluate(X_test,y_test, batch_size = batch_size)
    print('epoch {} test loss {}'.format(i,test_loss[i]))


np.save('test_loss.npy', test_loss)
train_loss = np.load('train_loss.npy')
val_loss = np.load('val_loss.npy')


# summarize history for loss
plt.plot(train_loss)
plt.plot(val_loss)
plt.plot(test_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation', 'Test'], loc='upper right')
plt.savefig('loss.png')
plt.show()

