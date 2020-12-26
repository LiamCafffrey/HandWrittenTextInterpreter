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
from input_preprocess import img_prep


neural_model = load_predict_neural()

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#3f87a6, #ebf8e1);
    color: black;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.subheader('Select the Page')

selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Processing', 'About')
    )

def welcome():

    page_bg_img = '''
    <style>
    .reportview-container {
        width: 100%;
        height: 100%;
        min-width: 100%;
        min-height: 100%;
        position: relative;
        }
    .reportview-container::before {
        background-image: url(https://www.criteo.com/wp-content/uploads/2017/11/17_ML_Thumbnails_Cheat-Sheet-1024x576.jpg);
        background-size: cover;
        content: "";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -2;
        opacity: 0.4;
        }
    .reportview-container::after {
        background-color: #3f87a6;
        content: "";
        display: block;
        position: absolute;
        top: 0px;
        left: 0px;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.4;
        }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title('Image Processing')

    st.subheader('A simple app that shows how powerfull a Neural Network can be. Upload an image and watch the magic happen..')

    st.image('../web_app_resources/welcome3.jpg',use_column_width=True)

def home():
    page_bg_img = '''
    <style>
    .reportview-container {
        width: 100%;
        height: 100%;
        min-width: 100%;
        min-height: 100%;
        position: relative;
        }
    .reportview-container::before {
        background-image: url(https://www.criteo.com/wp-content/uploads/2017/11/17_ML_Thumbnails_Cheat-Sheet-1024x576.jpg);
        background-size: cover;
        content: "";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -2;
        opacity: 0.4;
        }
    .reportview-container::after {
        background-color: #3f87a6;
        content: "";
        display: block;
        position: absolute;
        top: 0px;
        left: 0px;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.4;
        }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title('Text_interpreter')

    st.set_option('deprecation.showfileUploaderEncoding', False)



    img_file_buffer = st.file_uploader("Upload an image", type=["jpg"])

    st.markdown('When uploading an image, please be aware that the model needs the picture to be **one singular letter/char of the english alphabet** photographed in a **white backgroud** and written with a **black marker**.')

    if img_file_buffer != None:

        image = Image.open(io.BytesIO(img_file_buffer.read())).convert('RGB')
        inverted_image, image_predict_on = img_prep(image)
        c1,c2,c3,c4 = st.beta_columns((2,4,6,8))
        pred = interpreter(neural_model.predict_classes(image_predict_on))
        alphabet = 'A aB bC cD dE eF fG gH hI iJ jK kL lM mN nO oP pQ qR rS sT tU uV vW wX xY yZ z'
        num = '0123456789'
        if pred in alphabet:
            st.title(f'I predict your input to be the letter : {pred} :sunglasses:')
        elif pred in num:
            st.title(f'I predict your input to be the number : {pred} :sunglasses:')
        else:
            st.title('Please select a different file')


    else:
        st.warning('No image has been selected')


def about():
    page_bg_img = '''
    <style>
    .reportview-container {
        width: 100%;
        height: 100%;
        min-width: 100%;
        min-height: 100%;
        position: relative;
        }
    .reportview-container::before {
        background-image: url(https://www.criteo.com/wp-content/uploads/2017/11/17_ML_Thumbnails_Cheat-Sheet-1024x576.jpg);
        background-size: cover;
        content: "";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -2;
        opacity: 0.4;
        }
    .reportview-container::after {
        background-color: #3f87a6;
        content: "";
        display: block;
        position: absolute;
        top: 0px;
        left: 0px;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.4;
        }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)


    st.title('work in progress')

if selected_box == 'Welcome':
    welcome()
if selected_box == 'Image Processing':
    home()
if selected_box == 'About':
    about()
