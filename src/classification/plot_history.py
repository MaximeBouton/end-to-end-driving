###############################################################################
#    SCRIPT TO VISUALIZE TEST ERROR EVOLUTION
#
###############################################################################

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


### Some parameters
batch_size = 32
nb_classes = 3
nb_epoch = 40
data_augmentation = False

# input image dimensions
factor  = 8.0
img_rows, img_cols = int(480/factor), int(640/factor)
# the images are RGB
img_channels = 3

### Load DATA
dataset_name = '../data/resized_features' + str(int(factor)) + '.npy'
X = np.load(dataset_name)
y = np.load('../data/labels.npy')

# shuffle the data
np.random.seed(3)
p = np.random.permutation(len(y))
y = y[p]
X = X[p,:,:,:]

(X_train, y_train) = X[:int(.6*X.shape[0])], y[:int(.6*X.shape[0])]
(X_val,  y_val)    = X[int(.6*X.shape[0]):int(.72*X.shape[0])], y[int(.6*X.shape[0]):int(.72*X.shape[0])]
(X_test, y_test)   = X[int(.72*X.shape[0]):], y[int(.72*X.shape[0]):]
Y_test  = np_utils.to_categorical(y_test, nb_classes)

### Generate Model and evaluate for each weights


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_rows, img_cols,3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

### Compute test acc and loss

test_acc = np.zeros(30)
test_loss = np.zeros(30)

for i in range(30):

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_rows, img_cols,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # load pre-trained weights
    weights = 'tmp/weights.%02d'%i
    model.load_weights(weights)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                                        metrics=['accuracy'])

    test_loss[i],test_acc[i] = model.evaluate(X_test,Y_test, batch_size = batch_size)

np.save('test_loss.npy', test_loss)
np.save('test_acc.npy', test_acc)

## Load training log
train_acc  = np.load('train_acc.npy')
val_acc    = np.load('val_acc.npy')
train_loss = np.load('train_loss.npy')
val_loss   = np.load('val_loss.npy')

# summarize history for accuracy
plt.plot(train_acc)
plt.plot(val_acc)
plt.plot(test_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation', 'Test'], loc='lower right')
plt.savefig('accuracy.png')
plt.show()

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
