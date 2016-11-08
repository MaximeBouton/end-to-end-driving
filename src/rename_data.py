from shutil import copyfile
from os import listdir
from os.path import isfile, join
import math as m
import csv
from itertools import islice


srcPathLeft   = '../data/left_camera/'
dstPathLeft   = '../data/leftCamera/'

srcPathCenter = '../data/center_camera/'
dstPathCenter = '../data/centerCamera/'

srcPathRight  = '../data/right_camera/'
dstPathRight  = '../data/rightCamera/'

header = 'id,seq,time_stamp,steering_wheel_angle,steering_wheel_torque,speed'
f = open('../data/cleanSteering.csv','w')
f.write(header)
f.close()

imagesLeft   = [f for f in listdir(srcPathLeft) if isfile(join(srcPathLeft, f))]
imagesCenter = [f for f in listdir(srcPathCenter) if isfile(join(srcPathCenter, f))]
imagesRight  = [f for f in listdir(srcPathRight) if isfile(join(srcPathRight, f))]

imagesLeft   = sorted(imagesLeft)
imagesCenter = sorted(imagesCenter)
imagesRight  = sorted(imagesRight)

nLeft   = len(imagesLeft)
nCenter = len(imagesCenter)
nRight  = len(imagesRight)

if nLeft != nCenter or nCenter != nRight:
    raise Exception("different image counts (left - center - right)")

#print('left : {}'.format(nLeft))
#print('cntr : {}'.format(nCenter))
#print('rght : {}'.format(nRight))


nDigits = int(m.ceil(m.log(nLeft,10)))
sid = "%0" + str(nDigits) + "d"

lastR = 1
id = 0
for i in range(nLeft):
    id += 1
    
    timeL = imagesLeft[i].split('.')[0]
    timeC = imagesCenter[i].split('.')[0]
    timeR = imagesRight[i].split('.')[0]
    power = 7
    time = round((float(timeL) + float(timeC) + float(timeR))/3/pow(10,power))/pow(10,9-power)
        
    srcL = srcPathLeft + imagesLeft[i]
    srcC = srcPathCenter + imagesCenter[i]
    srcR = srcPathRight + imagesRight[i]
    
    name = sid % (id,)
    
    dstL = dstPathLeft + name + ".jpeg"
    dstC = dstPathCenter + name + ".jpeg"
    dstR = dstPathRight + name + ".jpeg"
    
    copyfile(srcL, dstL)
    copyfile(srcC, dstC)
    copyfile(srcR, dstR)

    f = open('../data/steering.csv','r')
    reader = csv.reader(f, delimiter=',')
    maxRow = 37977
    rm = -999
    rp = -999
    r = lastR
    for row in islice(reader,lastR,maxRow):
        rowTime = float(row[1])/pow(10,9)
        rp = row
        if rowTime > time:
            break
        rm = row
        r += 1
    f.close()
    lastR = r
    
    seq            = 0
#    rowTime        = 0
    steeringAngle  = 0
    steeringTorque = 0
    speed          = 0
    if rm == -999:
        seq            = int(rp[0])
#        rowTime        = float(rp[1])
        steeringAngle  = float(rp[2])
        steeringTorque = float(rp[4])
        speed          = float(rp[5])
    else:
        wm = (float(rp[1])-time*pow(10,9))/(float(rp[1]) - float(rm[1]))
        wp = (time*pow(10,9)-float(rm[1]))/(float(rp[1]) - float(rm[1]))
        seq            = round(wm*float(rm[0]) + wp*float(rp[0]),3)
#        rowTime        = wm*float(rm[1]) + wp*float(rp[1])
        steeringAngle  = wm*float(rm[2]) + wp*float(rp[2])
        steeringTorque = wm*float(rm[4]) + wp*float(rp[4])
        speed          = wm*float(rm[5]) + wp*float(rp[5])

    row = '\n' + name + ',' + str(seq) + ',' + str(time) + ',' + str(steeringAngle) + ',' + str(steeringTorque) + ',' + str(speed)
    f = open('../data/cleanSteering.csv','a')
    f.write(row)
    f.close()
    
    if id%100 == 0:
        print '#%05d/%05d [x]'%(id,nLeft)



