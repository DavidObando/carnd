import csv
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import os
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf

tf.python.control_flow_ops = tf

def make_model(n_classes, dropout_rate=0.5, regularizer_rate=0.0001):
    """
    Creates the keras model used for our network
    """
    model = Sequential()
    # Add a convolution with 32 filters, 3x3 kernel, and valid padding
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32, 32, 3)))
    # Add a max pooling of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a dropout
    model.add(Dropout(dropout_rate))
    # Add a ReLU activation layer
    model.add(Activation('relu'))

    # Add a convolution with 64 filters, 2x2 kernel, and valid padding
    model.add(Convolution2D(64, 2, 2, border_mode='valid'))
    # Add a max pooling of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a dropout
    model.add(Dropout(dropout_rate))
    # Add a ReLU activation layer
    model.add(Activation('relu'))

    # Add a flatten layer
    model.add(Flatten())
    # Add a fully connected layer
    #model.add(Dense(128, W_regularizer=l2(regularizer_rate), activity_regularizer=activity_l2(regularizer_rate)))
    model.add(Dense(128))
    # Add a ReLU activation layer
    model.add(Activation('relu'))
    # Add a fully connected layer
    #model.add(Dense(n_classes, W_regularizer=l2(regularizer_rate), activity_regularizer=activity_l2(regularizer_rate)))
    model.add(Dense(n_classes))
    # Add a ReLU activation layer
    model.add(Activation('softmax'))
    print(model.summary())

    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    return model

def resize_image(image_in, dimensions=(32,32)):
    """
    Resizes the input image to the specified dimensions
    """
    return cv2.resize(image_in, dimensions, interpolation = cv2.INTER_AREA)

def read_image_data(path):
    """
    Loads an image file, resizes it, returns it as a numpy array
    """
    return np.array(resize_image(mpimg.imread(path)))

def load_data():
    """
    Loads the training data
    """
    data_folder = "./data/"
    pickle_file = data_folder + "data.pickle"
    if os.path.exists(pickle_file):
        print('Loading data from pickle file...')
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
            images = pickle_data["images"]
            steering = pickle_data["steering"]
            del pickle_data
        return images, steering
    images = []
    steering = []
    with open(data_folder + "driving_log.csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #center,left,right,steering,throttle,brake,speed
            images.append(read_image_data(data_folder + row["center"].strip()))
            steering.append(float(row["steering"]))
            images.append(read_image_data(data_folder + row["left"].strip()))
            steering.append(float(row["steering"]))
            images.append(read_image_data(data_folder + row["right"].strip()))
            steering.append(float(row["steering"]))
    images = np.array(images)
    steering = np.array(steering)
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    "images": images,
                    "steering": steering,
                },
                pfile, pickle.DEFAULT_PROTOCOL)
    except Exception as e:
        print("Unable to save data to", pickle_file, ":", e)
        raise
    return images, steering

def normalize_minmax(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [-0.5, 0.5]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

X_train, y_train = shuffle(load_data())
n_train = len(X_train)
assert n_train == len(y_train)
print("X shape", X_train.shape)
print("y shape", y_train.shape)

X_normalized = normalize_minmax(X_train)

model = make_model(1)
history = model.fit(X_normalized, y_train, batch_size=256, nb_epoch=20, validation_split=0.2)
