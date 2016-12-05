import numpy as np


path  = './classification/'
# path  = './prelu_3_3_3_60/'
# path  = './prelu_3_2_3_60/'

#filename = 'train_loss'
filename = 'train_acc'
# filename = 'train45'
# filename = 'train_loss_prelu_3_2_3_6'
# filename = 'train_loss_prelu_3_2_3_6_cntd'

readname  = path + filename + '.npy'
writename = path + filename + '.dat'
A = np.load(readname)
L = len(A)
fid = open(writename, 'w')
for i in range(L):
    fid.write(str(A[i]) + '\n')
fid.close()


#filename = 'val_loss'
filename = 'val_acc'
# filename = 'val45'
# filename = 'val_loss_prelu_3_2_3_6'
# filename = 'val_loss_prelu_3_2_3_6_cntd'

readname  = path + filename + '.npy'
writename = path + filename + '.dat'
A = np.load(readname)
L = len(A)
fid = open(writename, 'w')
for i in range(L):
    fid.write(str(A[i]) + '\n')
fid.close()


#filename = 'test_loss'
filename = 'test_acc'
# filename = 'test45'
# filename = 'test_loss_prelu_3_2_3_6'
# filename = 'test_loss_prelu_3_2_3_6_cntd'

readname  = path + filename + '.npy'
writename = path + filename + '.dat'
A = np.load(readname)
L = len(A)
fid = open(writename, 'w')
for i in range(L):
    fid.write(str(A[i]) + '\n')
fid.close()
