import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from os import listdir

model_path ='model/'

num_classes = 200 # FILL IN NUMBER OF CLASSES
input_dim = (4096,1)



''' Use VGG network to extract features from input image '''

# Initialize VGG16 model using pretrained ImageNet weights without last 3 fc layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
print('Initializing VGG16 model with ImageNet weights...')

# Initialize two fc layers
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dense(4096, activation='relu'))

# Copy VGG layers in order to concatenate
VGG_model = Sequential()
for layer in base_model.layers:
    VGG_model.add(layer)

VGG_model.add(top_model)

# Make the layers untrainable
for layer in VGG_model.layers:
    layer.trainable = False
VGG_model.summary()



''' Extracts high level VGG features from a single image '''
def extract_VGG(img, model):
    img = cv2.imread(img)
    img_np = (np.array(img)).astype('float64')
    img_np = np.expand_dims(img_np, axis=0)
    img_data = preprocess_input(img_np)

    return model.predict(img_data)



def create_CNN():
    # Convolutional Neural Network Architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_dim))
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
            
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model



''' Loads previously saved model architecture + weights + optimizer state'''
def load_CNN(load_path, model_name):
    model = load_model(load_path+model_name)
    return model



# clf = create_CNN()

# Do some training

# clf.save(model_path+'CNN.h5')  # creates a HDF5 file, saves model to disk




