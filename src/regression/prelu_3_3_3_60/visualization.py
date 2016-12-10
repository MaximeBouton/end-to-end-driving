########### SCRIPT TO VISUALIZE CNN OUTPUT ################

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
### Generate Model 

batch_size = 32
nb_epoch = 45 
data_augmentation = False

# input image dimensions
factor  = 8.0
img_rows, img_cols = int(480/factor), int(640/factor)
# the images are RGB
img_channels = 3
fs = 3


## BUILD MODEL
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


model.load_weights('tmp/weights.25')

model.compile(loss='mean_squared_error', optimizer='adam')


## Compute the test acc and loss 
dataset_name = '../../../data/resized_features' + str(int(factor)) + '.npy'
X = np.load(dataset_name)
y = np.load('../regLabels.npy')

# zero mean labels
mean = y.mean()
std  = y.std()
y = (y - mean) / std

X /= 255.0

# shuffle the data 
np.random.seed(3)
p = np.random.permutation(len(y))
y = y[p]
X = X[p,:,:,:]


### Extract 1st Conv layer output 

#get_1st_layer_output = K.function([model.layers[0].input],[model.layers[0].output])

#FirstOutput = get_1st_layer_output([X[3739:3740,:,:,:]])[0]
#m,height,width, nFilters = FirstOutput.shape

# plot 8 filters output
import matplotlib.pyplot as plt

#f, ax = plt.subplots(8)
#k = nFilters/8 

#for i in range(8):
#    ax[i].imshow(FirstOutput[0,:,:,(i+1)*k-1])
#    ax[i].axis('off')

#plt.show()

# Plot 8 filters for each layer 
n_layers = 15 # stop befor FC layers

f, ax = plt.subplots(8,n_layers)

for i in range(n_layers):
    get_layer_output = K.function([model.layers[0].input],[model.layers[i].output])
    output = get_layer_output([X[3739:3740,:,:,:]])[0]
    m,height,width, nFilters = output.shape
    k = nFilters/8
    for j in range(8):
        ax[j,i].imshow(output[0,:,:,j])
        ax[j,i].axis('off')

f.savefig('reg_visualization.png')
f.show()

# Get output 
get_prediction = K.function([model.layers[0].input],[model.layers[-1].output])
ypred =  get_prediction([X[3739:3740,:,:,:]])[0]
print('Predicted output: {}'.format(ypred))
print('Real command: {}'.format(y[3739]))

