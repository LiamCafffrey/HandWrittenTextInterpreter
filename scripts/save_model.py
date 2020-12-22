import os
from tensorflow import keras

path = os.path.join('..','save_model','neural_model')

def save(model):
    model.save(path)
