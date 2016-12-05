import numpy as np


#path  = './classification/'
path  = './prelu_3_3_3_60/'

#filename = 'train_loss'
filename = 'train45'

readname  = path + filename + '.npy'
writename = path + filename + '.dat'
A = np.load(readname)
L = len(A)
fid = open(writename, 'w')
for i in range(L):
    fid.write(str(A[i]) + '\n')
fid.close()


#filename = 'val_loss'
filename = 'val45'

readname  = path + filename + '.npy'
writename = path + filename + '.dat'
A = np.load(readname)
L = len(A)
fid = open(writename, 'w')
for i in range(L):
    fid.write(str(A[i]) + '\n')
fid.close()


#filename = 'test_loss'
filename = 'test45'

readname  = path + filename + '.npy'
writename = path + filename + '.dat'
A = np.load(readname)
L = len(A)
fid = open(writename, 'w')
for i in range(L):
    fid.write(str(A[i]) + '\n')
fid.close()


