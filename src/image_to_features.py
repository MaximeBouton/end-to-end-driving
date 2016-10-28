import cv2 
import numpy as np

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
red = 20
w_,h_ = w/red,h/red

Phi = np.zeros((m,w_,h_,3))


print 'Start processing images'
for (i,j) in enumerate(bigRange):
    if i%100==0:
        print 'Processing image %i' %i
    imgId = '%05d' %(j+1)
    imgName = path+imgId+'.jpeg'
    image = cv2.imread(imgName)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for k in range(w_):
        for l in range(h_):
            cell = gray_image[l*red:(l+1)*red,k*red:(k+1)*red]         
            Phi[i,k,l,0] = np.min(cell)
            Phi[i,k,l,1] = np.max(cell)
            Phi[i,k,l,2] = np.mean(cell)
np.save('features.npy',Phi)

print 'Images saved!'

