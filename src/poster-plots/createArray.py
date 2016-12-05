import numpy as np

name = 'abcd'

#a = np.zeros((2,3,4))
#a[0,1,2] = 5
#a[1,2,1] = 2


a = np.ones(10)

for i in range(10):
    a[i] *= i

np.save(name, a)
