###############################################################################
#
# Script to resize images by factor 8 and output a feature vector
#
###############################################################################

import cv2
import numpy as np

# Location of the images
path = '../../data/centerCamera/'


# Remove  Outlayers

ranges = [(1,1590),(1844,4965),(5180,8878),(9985,11380),(11740,13340),(13710,14060),(14680,15120)]

bigRange = []

for r in ranges:
        bigRange+= range(r[0],r[1]+1)

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
    imgId = '%05d' %(j+1)
    imgName = path+imgId+'.jpeg'
    image = cv2.imread(imgName)
    # resize
    resized_img = cv2.resize(image, (w_,h_), interpolation = cv2.INTER_AREA)
    Phi[i,:,:,:] = resized_img

# Save feature matrix to file
fname = 'resized_fetures' + str(red) + '.npy'
np.save(fname,Phi)
