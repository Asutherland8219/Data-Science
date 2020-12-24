## https://www.datacamp.com/community/tutorials/tensorflow-tutorial ##

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import PIL
import glob
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pathlib import Path
from PIL import Image
from skimage.color import rgb2gray


# Test the tensorflow model by running a simple equation
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1, x2)

print(result)

# Create a tf dataset  from the two image data sets using the image_dataset_from_directory function


CATEGORIES = ['Positive', 'Negative']

training_data= []

def create_training_data():
        for category in CATEGORIES:
            files = glob.glob('C:/Users/asuth/Data Sets/Concrete_crack/*.jpg')
            for img in os.listdir(files):
                img_array = cv2.imread(os.files.join(files,img), cv2.IMREAD_GRAYSCALE)
                training_data.append()




# iterate through the names of contents of the folder

# negative set 
image_nocrack = []
files = glob.glob('C:/Users/asuth/Data Sets/Concrete_crack/Negative/*.jpg')
for myfile in files:
    image = cv2.imread (myfile)
    image_g = rgb2gray(image)
    image_nocrack.append (image_g)

print( 'no crack image shapes: ', np.array(image_nocrack).shape)

# positive set 
image_crack = []
files = glob.glob('C:/Users/asuth/Data Sets/Concrete_crack/Positive/*.jpg')
for myfile in files:
    image = cv2.imread (myfile)
    image_g = rgb2gray(image)
    image_crack.append (image_g)

print( 'crack image shapes: ', np.array(image_crack).shape)


# Create a total set to train, test, split.

full_data = np.concatenate((image_nocrack, image_crack))

print(np.array(full_data).shape)










