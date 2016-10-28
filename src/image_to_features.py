import cv2 
import numpy as np

path = '../../data/centerCamera/'

# size of the training set
m = 100
w,h = 640,480
red = 20
w_,h_ = w/red,h/red

Phi = np.zeros((m,w_,h_,3))

print 'Start processing images'
for i in range(m):
    imgId = '%05d' %(i+1)
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

