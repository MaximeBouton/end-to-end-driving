###################################################################################
#    Script to convert images from a Camera to features vectors
#    
#  Input: Folder with all the jpeg imgs well named
#  Output: - features.npy file with a 4 dimension matrix m x w x h x 3
#          - labels.npy file with a vector of size m containing the labels 
#  m = number of images 
#  w = width of the images
#  h = height of the images
#  3 : number of interesting values per pixels (rgb or min,max,av in grayscale)
#

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
red = 1 
w_,h_ = w/red,h/red

Phi = np.zeros((m,w_,h_,3))


# Start the processing
print 'Start processing images'
for (i,j) in enumerate(bigRange):
    if i%100==0:
        print 'Processing image %i' %i
    imgId = '%05d' %(j+1)
    imgName = path+imgId+'.jpeg'
    image = cv2.imread(imgName)
    # convert to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Loop through all the defined cells
    for k in range(w_):
        for l in range(h_):
            # for each cells extract information
            cell = gray_image[l*red:(l+1)*red,k*red:(k+1)*red]         
            Phi[i,k,l,0] = np.min(cell)
            Phi[i,k,l,1] = np.max(cell)
            Phi[i,k,l,2] = np.mean(cell)

# Save feature matrix to file
np.save('features.npy',Phi)

print 'Images saved!'

# Do the same for the steering
steeringFile = open('../../data/baselineTernaryClassification.csv','r')
nLabel = 0

# initialize vector
labels = np.zeros(m)
step = 0
# skip header
line = steeringFile.readline()

while line != '':
    if step%100==0:
        print 'Processing label %i' %step
    line = steeringFile.readline()
    line_i = line.split(',')
    steeringAngle = int(line_i[-1])
    labels[step] = steeringAngle
    step += 1

assert step==m # check that there are the good number of label
np.save('labels.npy',labels)

print 'Labels saved!'

steeringFile.close()


