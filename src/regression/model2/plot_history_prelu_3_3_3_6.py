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

# shuffle the data 
np.random.seed(3)
p = np.random.permutation(len(y))
y = y[p]
X = X[p,:,:,:]

mean = y.mean() 
std = y.std() 
y = (y-mean) / std


fr_tr = .6
fr_va = .12
## Repartition 

(X_train, y_train) = X[:int(fr_tr*X.shape[0])], y[:int(fr_tr*X.shape[0])]
(X_val,  y_val)    = X[int(fr_tr*X.shape[0]):int((fr_tr+fr_va)*X.shape[0])], y[int(fr_tr*X.shape[0]):int((fr_tr+fr_va)*X.shape[0])]
(X_test, y_test)   = X[int((fr_tr+fr_va)*X.shape[0]):], y[int((fr_tr+fr_va)*X.shape[0]):]

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
model.add(PReLU(init='zero', weights=None))
model.add(Convolution2D(32, fs, fs))
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


### Compute test loss

#train_loss = np.zeros(nb_epoch)
#val_loss   = np.zeros(nb_epoch)
test_loss  = np.zeros(nb_epoch)

for i in range(nb_epoch):
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

    weights = 'tmp3/weights.%02d'%i 
    model.load_weights(weights)

    model.compile(loss='mean_squared_error', optimizer='adam')

    #train_loss[i]  = model.evaluate(X_train,y_train, batch_size = batch_size)
    #val_loss[i]    = model.evaluate(X_val,y_val, batch_size = batch_size)
    test_loss[i]   = model.evaluate(X_test,y_test, batch_size = batch_size)
    print('test loss at epoch {} : {}'.format(i,test_loss[i]))



#np.save('train_loss_prelu_3_2_3_6.npy', train_loss)
#np.save('val_loss_prelu_3_2_3_6.npy', val_loss)
np.save('test_loss30.npy', test_loss)

train_loss60 = np.load('train60.npy')
val_loss60   = np.load('val60.npy')
train_loss30 = np.load('train_loss.npy')
val_loss30   = np.load('val_loss.npy')
#test_loss1  = np.load('test_loss15.npy')
test_loss60  = np.load('test60.npy')

train_loss = np.concatenate([train_loss60,train_loss30])
val_loss   = np.concatenate([val_loss60, val_loss30])
test_loss  = np.concatenate([test_loss60,test_loss])

np.save('train90.npy',train_loss)
np.save('val90.npy',val_loss)
np.save('test90.npy',test_loss)

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

