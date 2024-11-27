# Helping source: https://keras.io/examples/vision/visualizing_what_convnets_learn/
# Helping source: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

from keras.models import Sequential
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import load_model
from keras import backend as K

import numpy as np
import argparse
import cv2
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

# ---- load data -------------------------------------------------------

# path to training images
train_path = 'train'

# path to validation images
test_path = 'test'

# images to be resized to (image_dim) x (image_dim)
image_dim = 128

x_train = []
y_train = []
x_test = []
y_test = []

# path to csv files
csvFile_train='output_train.csv'
csvFile_test='output_test.csv'

train_read=pd.read_csv(csvFile_train)
test_read=pd.read_csv(csvFile_test)

# load training data
i_train = 0
for filename in next(os.walk(train_path))[2]:
        # full path is path to filename + '/' + filename
        image = cv2.imread(''.join([train_path, '/', filename]))
        # append resized image
        x_train.append(cv2.resize(image, (image_dim, image_dim)))
        # construct training output data
        x_train_value=train_read['x'].values[i_train]
        y_train_value=train_read['y'].values[i_train]
        z_train_value=train_read['z'].values[i_train]
        r_train_value=train_read['r'].values[i_train]
        p_train_value=train_read['p'].values[i_train]
        q_train_value=train_read['q'].values[i_train]
        joint1_train_value=train_read['joint1'].values[i_train]
        joint2_train_value=train_read['joint2'].values[i_train]
        joint3_train_value=train_read['joint3'].values[i_train]
        joint4_train_value=train_read['joint4'].values[i_train]
        y_train_vector = [x_train_value, y_train_value, z_train_value, r_train_value, p_train_value, q_train_value, joint1_train_value, joint2_train_value, joint3_train_value, joint4_train_value]
        y_train.append(y_train_vector)
        i_train=i_train+1

# load test data
i_test = 0
for filename in next(os.walk(test_path))[2]:
        # full path is path to filename + '/' + filename
        image = cv2.imread(''.join([test_path, '/', filename]))
        # append resized image
        x_test.append(cv2.resize(image, (image_dim, image_dim)))
        # construct test output data
        x_test_value=test_read['x'].values[i_test]
        y_test_value=test_read['y'].values[i_test]
        z_test_value=test_read['z'].values[i_test]
        r_test_value=test_read['r'].values[i_test]
        p_test_value=test_read['p'].values[i_test]
        q_test_value=test_read['q'].values[i_test]
        joint1_test_value=test_read['joint1'].values[i_test]
        joint2_test_value=test_read['joint2'].values[i_test]
        joint3_test_value=test_read['joint3'].values[i_test]
        joint4_test_value=test_read['joint4'].values[i_test]
        y_test_vector = [x_test_value, y_test_value, z_test_value, r_test_value, p_test_value, q_test_value, joint1_test_value, joint2_test_value, joint3_test_value, joint4_test_value]
        y_test.append(y_test_vector)
        i_test=i_test+1  

# convert data to NumPy array of floats
x_train = np.array(x_train, np.float32)
x_test = np.array(x_test, np.float32)

# ---- define metrics ----
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# ---- load model -----
model = load_model('regression_early_stopping_1.h5' , custom_objects={"coeff_determination": coeff_determination })

# retrieve weights from the second hidden layer
filters, biases = model.layers[3].get_weights()


# normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)


# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()





