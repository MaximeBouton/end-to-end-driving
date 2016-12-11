###############################################################################
#                                                                             #
#            write numpy array to .dat file for matlab                        #
#                                                                             #
###############################################################################

import numpy as np

### Load np array
fname = 'test-predictions'
a = np.load(fname+'.npy')

### open .dat file

f = open(fname+'.dat',w)
