import matplotlib.pyplot as plt
import csv
import numpy as np

f = open('../data/baselineTernaryClassification.csv','r')

f.readline()

cmds = []
ids = []

line = f.readline()

while line!='':
    line_i = line.split(',')
    ids.append(int(line_i[0]))
    cmds.append(int(line_i[1]))
    line = f.readline()

print len(ids)
print len(cmds)

plt.plot(cmds)
plt.show()
