
#####################################################################################
#   5-LABELING.PY                                                                   #
#####################################################################################

# Reading data from input file and making QUINARY classification based based on the 
# steering angle into the five categories
# - strongly left,
# - slightly left,
# - straight,
# - slightly right, or
# - strongly right
# driving.


#####################################################################################
#   INPUT AND OUTPUT FILE                                                           #
#####################################################################################

fname = '../../data/cleanSteering.csv'
fr = open(fname,'r')
fr.readline()

fname = '../../data/baselineQuinaryClassification.csv'
fw = open(fname,'w')
fw.write('id,classification')


#####################################################################################
#   CLASSIFICATION PARAMETERS                                                       #
#####################################################################################

mean = 0.023906183832534
std  = 0.011722130776126
f1   = 4.0/3
f2   = 2.0/3


#####################################################################################
#   COUNTING LABELS                                                                 #
#####################################################################################

countLL = 0
countL  = 0
countC  = 0
countR  = 0
countRR = 0


#####################################################################################
#   DOING THE LABELING                                                              #
#####################################################################################

classification = 0
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
    if steeringAngle >= mean + f1 * std:
        classification = 2
        countLL += 1
    elif steeringAngle >= mean + f2 * std:
        classification = 1
        countL  += 1
    elif steeringAngle <= mean - f1 * std:
        classification = -2
        countRR += 1
    elif steeringAngle <= mean - f2 * std:
        classification = -1
        countR  += 1
    else:
        classification = 0
        countC += 1
    fw.write('\n' + id + ',' + str(classification))
    line = fr.readline()


#####################################################################################
#   PUBLISHING STATS                                                                #
#####################################################################################

totalCount = countLL + countL + countC + countR + countRR
print('total sample # = {}'.format(totalCount))
print(' >>  strongly left : {} [{}%]'.format(countLL,round(float(countLL)/totalCount*100,1)))
print(' >>  slightly left : {} [{}%]'.format(countL ,round(float(countL )/totalCount*100,1)))
print(' >>       straight : {} [{}%]'.format(countC ,round(float(countC )/totalCount*100,1)))
print(' >> slightly right : {} [{}%]'.format(countR ,round(float(countR )/totalCount*100,1)))
print(' >> strongly right : {} [{}%]'.format(countRR,round(float(countRR)/totalCount*100,1)))


#####################################################################################
#   FINISHING                                                                       #
#####################################################################################

fr.close()
fw.close()


