import numpy as np
from sklearn import datasets, svm, metrics
import math

## Load the dataset

X = np.load('features.npy')
Y = np.load('labels.npy')
m = len(Y)

# reshape the data
X = X.reshape(m,-1)
Y = Y.reshape(m,-1)

# separate training
tra =int(math.floor(0.8*m)) # training set size
val =int(math.floor(0.2*m)) # validation set size
tes =int( m - tra - val )    # test set size

Xtra = X[:tra]
Ytra = Y[:tra]

Xval = X[tra:tra+val]
Yval = Y[tra:tra+val]

Xtes = X[tra+val:]
Ytes = Y[tra+val:]


# Create a supprot vector classifier with RGB kernel
classifier = svm.SVC(gamma=0.0001,verbose=True)

# Learn
classifier.fit(Xtra, Ytra)

# predict
predicted = classifier.predict(Xval)

print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(Yval, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Yval, predicted))


