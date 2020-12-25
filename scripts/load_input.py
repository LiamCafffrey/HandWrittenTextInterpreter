from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def load_image(filename):

    img = load_img(filename, grayscale=True, target_size=(28, 28))

    img = img_to_array(img)

    img = img.astype('float32')
    img = img / 255.0
    return img
