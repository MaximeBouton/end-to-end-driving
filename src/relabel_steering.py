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

# some stats
totalL = 0
totalC = 0
totalR = 0

# loop through all samples from input and write classification to output
line = fr.readline()
while line != '':
    line_i = line.split(',')
    id = line_i[0]
    steeringAngle = float(line_i[3])
    if steeringAngle >= thetaL:
        classification = 1
        totalL += 1
    elif steeringAngle <= thetaR:
        classification = -1
        totalR += 1
    else:
        classification = 0
        totalC += 1
    fw.write('\n' + id + ',' + str(classification))
    line = fr.readline()

# finish
fr.close()
fw.close()

# publish stats
totalCount = totalL+totalC+totalR
print('total sample # : {}'.format(totalCount))
print(' >>       left : {} [{}%]'.format(totalL,round(float(totalL)/totalCount*100,1)))
print(' >>   straight : {} [{}%]'.format(totalC,round(float(totalC)/totalCount*100,1)))
print(' >>      right : {} [{}%]'.format(totalR,round(float(totalR)/totalCount*100,1)))


