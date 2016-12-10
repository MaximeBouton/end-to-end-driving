###############################################################################
#          PLOT STEERING LABELS AND FIND THE INTERESTING INTERVALS 
#
###############################################################################


import numpy as np

## Count number of samples
steeringFile = open('../../data/regressionSteering2.csv')

#skip headers
line = steeringFile.readline()

m = 0
while line!='':
    line= steeringFile.readline()
    m+=1
    print 'counting %d' %m

steeringFile.close()

print('Number of samples = {}'.format(m))


## save them to numpy array

y = np.zeros(m)
step = 0

steeringFile = open('../../data/regressionSteering2.csv')
#skip headers
line = steeringFile.readline()

while True:
    if step%100==0:
        print 'processing label %i' %step
    line = steeringFile.readline()
    if line=='':
        break 
    line_i = line.split(',')
    steeringAngle = float(line_i[-1])
    y[step] = steeringAngle
    step += 1 

assert step==m # check that there are the good number of labels
np.save('regLabels2',y)

print 'Labels saved!'

steeringFile.close()

### PLOT THE DATA 
import matplotlib.pyplot as plt

plt.plot(y)
plt.title('steering command')
plt.show()


