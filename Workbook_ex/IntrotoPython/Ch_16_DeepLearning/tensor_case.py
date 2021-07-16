from token import ISTERMINAL
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

import matplotlib.pyplot as pyplot
import seaborn as sns 
sns.set(font_scale=2)

''' Display some of the pictures '''
import numpy as np 
index = np.random.choice(np.arange(len(X_train)), 24, replace=False)
figure, axes = pyplot.subplots(nrows=4, ncols=6, figsize=(16, 9))

for item in zip(axes.ravel(), X_train[index], y_train[index]):
    axes, image, target = item 
    axes.imshow(image, cmap=pyplot.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
pyplot.tight_layout()
pyplot.show()

''' Prepare the data '''
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

#normalize the data 
X_train = X_train.astype('float32')/ 255
X_test = X_test.astype('float32')/255

''' One hot encoding the data; takes all the images and classifies them as arrays of 1s and 0s '''
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)

# Verify the changes all at once
print(y_train[0])

y_test = to_categorical(y_test)

''' Creating the neural network '''
from tensorflow.keras.models import Sequential
cnn = Sequential()

# adding layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
