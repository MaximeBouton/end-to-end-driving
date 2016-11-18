# Some raw output

## 3 classes classification

### Basic test

```
Train on 8539 samples, validate on 3660 samples
Epoch 1/10
8539/8539 [==============================] - 335s - loss: 0.4914 - acc: 0.8644 - val_loss: 1.7789 - val_acc: 0.3339
Epoch 2/10
8539/8539 [==============================] - 339s - loss: 0.4709 - acc: 0.8667 - val_loss: 1.6644 - val_acc: 0.3339
Epoch 3/10
8539/8539 [==============================] - 338s - loss: 0.4505 - acc: 0.8667 - val_loss: 2.1087 - val_acc: 0.3339
Epoch 4/10
8539/8539 [==============================] - 963s - loss: 0.3642 - acc: 0.8814 - val_loss: 3.4155 - val_acc: 0.3339
Epoch 5/10
8539/8539 [==============================] - 342s - loss: 0.2646 - acc: 0.9099 - val_loss: 4.3520 - val_acc: 0.3421
Epoch 6/10
8539/8539 [==============================] - 335s - loss: 0.1729 - acc: 0.9377 - val_loss: 3.6502 - val_acc: 0.3667
Epoch 7/10
8539/8539 [==============================] - 334s - loss: 0.1167 - acc: 0.9594 - val_loss: 4.4748 - val_acc: 0.3727
Epoch 8/10
8539/8539 [==============================] - 334s - loss: 0.0907 - acc: 0.9670 - val_loss: 4.3579 - val_acc: 0.3713
Epoch 9/10
8539/8539 [==============================] - 335s - loss: 0.0720 - acc: 0.9737 - val_loss: 5.3931 - val_acc: 0.3421
Epoch 10/10
8539/8539 [==============================] - 336s - loss: 0.0658 - acc: 0.9763 - val_loss: 5.1277 - val_acc: 0.3475
```

###  With data shuffling


```
Using TensorFlow backend.
X_train shape: (8539, 60, 80, 3)
8539 train samples
3660 test samples
total training sample # : 8539
 >>       left : 1927 [22.6%]
 >>   straight : 6023 [70.5%]
 >>      right : 589 [6.9%]
total validation sample # : 3660
 >>       left : 822 [22.5%]
 >>   straight : 2600 [71.0%]
 >>      right : 238 [6.5%]
/usr/local/lib/python2.7/dist-packages/keras/utils/np_utils.py:23: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  Y[i, y[i]] = 1.
Not using data augmentation.
Train on 8539 samples, validate on 3660 samples
Epoch 1/10
8539/8539 [==============================] - 422s - loss: 0.7827 - acc: 0.7029 - val_loss: 0.7451 - val_acc: 0.7104
Epoch 2/10
8539/8539 [==============================] - 375s - loss: 0.6458 - acc: 0.7460 - val_loss: 0.4613 - val_acc: 0.8243
Epoch 3/10
8539/8539 [==============================] - 402s - loss: 0.4655 - acc: 0.8227 - val_loss: 0.3085 - val_acc: 0.9052
Epoch 4/10
8539/8539 [==============================] - 389s - loss: 0.3035 - acc: 0.8877 - val_loss: 0.1865 - val_acc: 0.9380
Epoch 5/10
8539/8539 [==============================] - 384s - loss: 0.2092 - acc: 0.9246 - val_loss: 0.1290 - val_acc: 0.9541
Epoch 6/10
8539/8539 [==============================] - 384s - loss: 0.1499 - acc: 0.9485 - val_loss: 0.0976 - val_acc: 0.9631
Epoch 7/10
8539/8539 [==============================] - 397s - loss: 0.1244 - acc: 0.9567 - val_loss: 0.0984 - val_acc: 0.9683
Epoch 8/10
8539/8539 [==============================] - 435s - loss: 0.1052 - acc: 0.9646 - val_loss: 0.0648 - val_acc: 0.9760
Epoch 9/10
8539/8539 [==============================] - 739s - loss: 0.0910 - acc: 0.9692 - val_loss: 0.0732 - val_acc: 0.9738
Epoch 10/10
8539/8539 [==============================] - 632s - loss: 0.0842 - acc: 0.9706 - val_loss: 0.0628 - val_acc: 0.9727
```



## 5 classes classification

