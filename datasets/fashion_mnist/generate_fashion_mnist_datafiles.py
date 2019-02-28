"""
Verified to work with Tensorflow 1.9

For MNIST10, writes two seperate `.mat` files containing the training and
test set respectively.

Pixel values are stored as uint8s (0-255); labels are zero-indexed.
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist 
import numpy as np
import scipy.io as sio

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# pad final dimension of dataset
x_train_f = np.expand_dims(x_train, 3)
x_test_f = np.expand_dims(x_test, 3)

sio.savemat('fashion_mnist_int_train.mat', {'images': x_train_f, 'labels': y_train})
sio.savemat('fashion_mnist_int_test.mat', {'images': x_test_f, 'labels': y_test})
