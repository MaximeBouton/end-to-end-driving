# reading data from input file and making ternary classification based based on the steering angle
# into left, straight, or right driving

# input and output file
fname = '../../data/cleanSteering.csv'
fr = open(fname,'r')
fr.readline()

fname = '../../data/baselineTernaryClassification.csv'
fw = open(fname,'w')
fw.write('id,classification')


# steering angle thresholds for left / straight / right
thetaL = .05    # don't know any more which values make sense
thetaR = -.05   # script runs in < 1 sec so we can adjust easily
classification = 0

# loop through all samples from input and write classification to output
line = fr.readline()
while line != '':
    line_i = line.split(',')
    id = line_i[0]
    steeringAngle = float(line_i[3])
    if steeringAngle >= thetaL:
        classification = 1
    elif steeringAngle <= thetaR:
        classification = -1
    else:
        classification = 0
    fw.write('\n' + id + ',' + str(classification))
    line = fr.readline()

# finish
fr.close()
fw.close()
