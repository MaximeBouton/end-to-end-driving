###############################################################################
#
#   EXTRACT  GOOD TEST SET
#
###############################################################################
import numpy as np

select = range(57820,59830) + range(63470,68050) #TODO select a range
m = len(select) # number of samples
y = np.zeros(m)
#####################################################################################
#   INPUT AND OUTPUT FILE                                                           #
#####################################################################################

fname = '../../data/regressionSteering2.csv'
fr = open(fname,'r')
fr.readline()

fname = '../../data/clean-test.csv'
fw = open(fname,'w')
fw.write('id,angle')

#####################################################################################
#   DOING THE LABELING                                                              #
#####################################################################################

line = fr.readline()
count = 0
ind = 0

while line != '':
    count += 1
    if not count in select:
        line = fr.readline()
        continue
    line_i = line.split(',')
    id = line_i[0]
    steeringAngle = float(line_i[1])
    y[ind] = steeringAngle
    ind+=1
    fw.write('\n' + id + ',' + str(steeringAngle))
    line = fr.readline()

#####################################################################################
#   CLOSE FILES  AND SAVE                                                                   #
#####################################################################################

fr.close()
fw.close()

np.save('test_labels.npy',y)
