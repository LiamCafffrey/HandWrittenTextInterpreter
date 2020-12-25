import pandas as pd
import streamlit as st
import numpy as np
import scripts
import cv2
import os
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from load_model import load_predict_neural
from tempfile import NamedTemporaryFile
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from load_input import load_image
from model_interpreter import interpreter
from PIL import Image, ImageOps

def img_prep(img):

        image_invert = ImageOps.invert(img) #invert image

        image_sized= image_invert.resize((28, 28)) #resizing image

        image_array = np.array(image_sized) #resizing image array

        image_black_white = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        image_black_white = np.expand_dims(image_black_white, 2) #these last 3 steps ensure that the image is only black and white

        image_black_white = image_black_white.astype('float32') #ensuring data type is float32 just how the model was trained

        image_normalized = image_black_white / 255.0 #normalizing the values of the image

        image_ready = image_normalized.reshape(1, 28, 28, 1) #adding the extra dimension so our model can predict on

        return image_invert,image_ready


