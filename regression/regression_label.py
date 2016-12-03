
#####################################################################################
#   regression_label.py                                                             #
#####################################################################################

# Reading data from input file and write the steering angle to a file

#####################################################################################
#   INPUT AND OUTPUT FILE                                                           #
#####################################################################################

fname = '../../data/cleanSteering.csv'
fr = open(fname,'r')
fr.readline()

fname = '../../data/regressionSteering.csv'
fw = open(fname,'w')
fw.write('id,angle')



#####################################################################################
#   DOING THE LABELING                                                              #
#####################################################################################

line = fr.readline()
count = 0
select = range(1,1590+1) + range(1844,4965+1) + range(5180,8878+1) + range(9985,11380+1) + range(11740,13340+1) + range(13710,14060+1) + range(14680,15120+1)
while line != '':
    count += 1
    if not count in select:
        line = fr.readline()
        continue
    line_i = line.split(',')
    id = line_i[0]
    steeringAngle = float(line_i[3])
    fw.write('\n' + id + ',' + str(steeringAngle))
    line = fr.readline()


#####################################################################################
#   CLOSE FILES                                                                     #
#####################################################################################

fr.close()
fw.close()


#####################################################################################
#   WRITE TO NUMPY ARRAY                                                            #
#####################################################################################

import numpy as np

m = len(select) # number of samples 

steeringFile = open('../../data/regressionSteering.csv')

# pre-allocate array 
labels = np.zeros(m)
step = 0
# skip header 
line = steeringFile.readline()
# run through file 
while True:
    if step%100==0:
        print 'Processing label %i' %step
    line = steeringFile.readline()
    if line=='':
        break
    line_i = line.split(',')
    steeringAngle = float(line_i[-1])
    labels[step] = steeringAngle
    step += 1

assert step==m # check that there are the good number of label
np.save('regLabels.npy',labels)

print 'Labels saved!'

steeringFile.close()


