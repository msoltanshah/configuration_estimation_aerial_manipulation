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
from keras import Model
import tensorflow as tf
from keras import preprocessing

import numpy as np
import argparse
import cv2
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from IPython.display import Image, display

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

# The dimensions of our input image
img_width = 250
img_height = 250
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = "conv2d_3" 

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = Model(inputs=model.inputs, outputs=layer.output)

def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

# loss, img = visualize_filter(0)
# preprocessing.image.save_img("0.png", img)

# display(Image("0.png"))

# Compute image inputs that maximize per-filter activations
# for the first 64 filters of our target layer
all_imgs = []
for filter_index in range(64):
    print("Processing filter %d" % (filter_index,))
    loss, img = visualize_filter(filter_index)
    all_imgs.append(img)

# Build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# Fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img = all_imgs[i * n + j]
        stitched_filters[
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j : (cropped_height + margin) * j
            + cropped_height,
            :,
        ] = img
preprocessing.image.save_img("stiched_filters.png", stitched_filters)

