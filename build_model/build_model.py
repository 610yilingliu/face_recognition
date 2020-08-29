import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import models , optimizers , losses
import tensorflow.keras.backend as K
from PIL import Image

def load_img(img_path):
    """
    :type img_path: string
    :rtype numpy.array
    """
    img = Image.open(img_path)
    return np.array(img)/255

def np_forSia(df, h, w):
    """
    :type df: pandas.DataFrame
    :type h, w: int(height and width of single image)
    """
    nimg = df.shape[0]
    xnp = np.zeros([nimg, 2, 1, h, w])
    ynp = np.zeros(nimg)
    for i in range(nimg):
        img1 = load_img(df.iloc[i, 0])
        img2 = load_img(df.iloc[i, 1])
        xnp[i, 0, 0, :, :] = img1
        xnp[i, 1, 0, :, :] = img2
        ynp[i] = df.iloc[i, 2]

    pic1s = tf.transpose(xnp[:, 0], [0, 2, 3, 1])
    pic2s = tf.transpose(xnp[:, 1], [0, 2, 3, 1])
    dim_x_for_tf = (xnp.shape[3], xnp.shape[4], xnp.shape[2])

    return pic1s, pic2s, ynp, dim_x_for_tf

def out_shape(shapes_pack):
    s1, s2 = shapes_pack
    return(s1[0], 1)

def loss_def(y_true, y_pred):
    margin = 1
    res = tf.keras.backend.mean(y_true * tf.keras.backend.square(y_pred) + (1 - y_true) * tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0)))
    return res

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


class Sia_net:
    def __init__(self, input_shape):
        self.losses = []
        kernel_size1 = (2, 2)
        kernel_size2 = (3, 3)
        pool_size = (3, 3)
        strides = 1
        model = tf.keras.Sequential()
        # convolution layer 1
        model.add(Conv2D(filters = 16, kernel_size = kernel_size2, strides = 1, activation = "relu", input_shape = input_shape, data_format = "channels_last", padding = "valid"))
        model.add(MaxPool2D(pool_size = pool_size))
        model.add(Dropout(0.25))
        # convolution layer 2
        model.add(Conv2D(filters = 32, kernel_size = kernel_size1, strides = 1, padding = "valid"))
        model.add(MaxPool2D(pool_size = pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.1))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # deep face mentioned that there are 67 points to detect on a human face
        model.add(Dense(64, activation='relu'))

        input_x1 = Input( shape=input_shape )
        input_x2 = Input( shape=input_shape )

        output_x1 = model( input_x1 )
        output_x2 = model( input_x2 )

        distance_euclid = Lambda( lambda tensors : K.abs(tensors[0] - tensors[1]))([output_x1 , output_x2])
        outputs = Dense(1 , activation = "sigmoid") (distance_euclid)
        self.__model = models.Model( [ input_x1 , input_x2 ] , outputs )
        self.__model.compile( loss=losses.binary_crossentropy , optimizer=optimizers.Adam(lr=0.0001))

    def fit(self, X, Y, hyperparameters):
        initial_time = time.time()
        hist = self.__model.fit( X, Y,
						 batch_size=hyperparameters[ 'batch_size' ] ,
						 epochs=hyperparameters['epochs'],
                         verbose = hyperparameters['verbose'] if hyperparameters['verbose'] else 0,
						 )
        self.losses.append(hist.history)
        final_time = time.time()
        eta = ( final_time - initial_time )
        time_unit = 'seconds'
        if eta >= 60 :
            eta = eta/60
            time_unit = 'minutes'
        print(self.__model.summary())
        print( 'Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters[ 'epochs' ] , eta , time_unit ))

    def evaluate(self , test_X , test_Y):
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, X):
        predictions = self.__model.predict(X)
        return predictions

    def summary(self):
    	self.__model.summary()

    def save_model(self , file_path):
        self.__model.save(file_path)

    def load_model(self , file_path ):
        self.__model = models.load_model(file_path)

def compute_accuracy(predictions, labels):
    """
    Could be optimized by multiprocessing but not convenient on Windows.
    OpenCV in C/C++ is better to do this
    """
    loss = 0
    length = len(predictions)
    pred_array = predictions.ravel()
    for i in range(length):
        if abs(predictions[i] - labels[i]) > 0.5:
            loss += 1
    return loss/length