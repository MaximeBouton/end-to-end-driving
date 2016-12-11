###############################################################################
#
# Script to resize images by factor 8 and output a feature vector
#
###############################################################################

import cv2
import numpy as np

# Location of the images
path = '../../data/centerCamera2/'


# Remove  Outlayers

bigRange = range(57820,59830) + range(63470,68050)

#size of the training set

m = len(bigRange)
print m
w,h = 640,480
red = 8  # resizing factor
w_,h_ = w/red,h/red

Phi = np.zeros((m,h_,w_,3))

# Start the processing
print 'Start processing images'

for (i,j) in enumerate(bigRange):
    if i%100==0:
        print 'Processing image %i' %i
    imgId = '%06d' %(j+1)
    imgName = path+imgId+'.jpeg'
    image = cv2.imread(imgName)
    # resize
    resized_img = cv2.resize(image, (w_,h_), interpolation = cv2.INTER_AREA)
    Phi[i,:,:,:] = resized_img

# Save feature matrix to file
fname = 'test_features' + str(red) + '.npy'
np.save(fname,Phi)
