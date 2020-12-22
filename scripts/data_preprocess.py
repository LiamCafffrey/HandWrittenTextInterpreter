import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

def ready_data():

    train_data = pd.read_csv('../raw_data/emnist-byclass-train.csv')
    test_data = pd.read_csv('../raw_data/emnist-byclass-test.csv')

    x_train = train_data.iloc[:,1:]
    y_train = train_data.iloc[:,0]

    x_test = test_data.iloc[:,1:]
    y_test = test_data.iloc[:,0]

    del train_data
    del test_data

    x_train = np.asarray(x_train)
    x_train = np.apply_along_axis(rotate, 1, x_train)

    x_test = np.asarray(x_test)
    x_test = np.apply_along_axis(rotate, 1, x_test)

    x_train = x_train/255
    x_test = x_test/255

    x_train = tf.keras.backend.expand_dims(x_train, axis=-1)
    x_test = tf.keras.backend.expand_dims(x_test, axis=-1)

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=62)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=62)

    return x_train, x_test, y_train_cat, y_test_cat
