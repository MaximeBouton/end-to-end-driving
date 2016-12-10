'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np

batch_size = 32
nb_classes = 3
nb_epoch = 30
data_augmentation = False

# input image dimensions
factor  = 8.0
img_rows, img_cols = int(480/factor), int(640/factor)
# the images are RGB
img_channels = 3

# the data, shuffled and split between train val and test sets
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
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')



# Stats of the training data 
train_R = 0
train_L = 0
train_S = 0

for e in y_train:
    if e==0:
        train_S += 1
    elif e==1:
        train_L += 1
    elif e==-1:
        train_R += 1

total = train_R+train_L+train_S 
assert total == len(y_train)

# publish stats
print('total training sample # : {}'.format(total))
print(' >>       left : {} [{}%]'.format(train_L,round(float(train_L)/total*100,1)))
print(' >>   straight : {} [{}%]'.format(train_S,round(float(train_S)/total*100,1)))
print(' >>      right : {} [{}%]'.format(train_R,round(float(train_R)/total*100,1)))

# Stats of the validation data 
val_R = 0
val_L = 0
val_S = 0

for e in y_val:
    if e==0:
        val_S += 1
    elif e==1:
        val_L += 1
    elif e==-1:
        val_R += 1

total = val_R+val_L+val_S
assert total == len(y_val)

# publish stats
print('total validation sample # : {}'.format(total))
print(' >>       left : {} [{}%]'.format(val_L,round(float(val_L)/total*100,1)))
print(' >>   straight : {} [{}%]'.format(val_S,round(float(val_S)/total*100,1)))
print(' >>      right : {} [{}%]'.format(val_R,round(float(val_R)/total*100,1)))

# Stats of the test data 
test_R = 0
test_L = 0
test_S = 0

for e in y_test:
    if e==0:
        test_S += 1
    elif e==1:
        test_L += 1
    elif e==-1:
        test_R += 1

total = test_R+test_L+test_S
assert total == len(y_test)

# publish stats
print('total test sample # : {}'.format(total))
print(' >>       left : {} [{}%]'.format(test_L,round(float(test_L)/total*100,1)))
print(' >>   straight : {} [{}%]'.format(test_S,round(float(test_S)/total*100,1)))
print(' >>      right : {} [{}%]'.format(test_R,round(float(test_R)/total*100,1)))


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val   = np_utils.to_categorical(y_val, nb_classes)
Y_test  = np_utils.to_categorical(y_test, nb_classes)

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

# Define a Callback function to save the weight and measure test error

checkpointer = ModelCheckpoint(filepath='tmp/weights.{epoch:02d}', verbose=1, save_best_only=False)

tfboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)



# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_val   = X_val.astype('float32')
X_test  = X_test.astype('float32')

X_train /= 255
X_val   /= 255
X_test  /= 255

print('Not using data augmentation.')
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    validation_data=(X_val, Y_val),
                    shuffle=True,
                    callbacks = [checkpointer,tfboard])

#### MODEL EVALUATION

testLoss, testAcc = model.evaluate(X_test,Y_test,
	       batch_size = batch_size)

print('Testing Loss = {}, accuracy = {}'.format(testLoss,testAcc))


#### save training stats

import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('loss.png')
plt.show()

np.save('train_acc.npy',history.history['acc'])
np.save('val_acc.npy',history.history['val_acc'])
np.save('train_loss.npy',history.history['loss'])
np.save('val_loss.npy',history.history['val_loss'])

