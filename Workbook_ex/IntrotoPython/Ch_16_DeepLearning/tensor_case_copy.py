from token import ISTERMINAL
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.metrics import accuracy

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
cnn.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

# add pooling layer 
cnn.add(MaxPooling2D(pool_size=(2,2)))

#2nd pooling layer 
cnn.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

# flatten the results 
cnn.add(Flatten())

# add dense layer
cnn.add(Dense(units=128, activation='relu'))

# add another dense layer
cnn.add(Dense(units=10, activation='softmax'))

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
''' Prep the model for tensorflow '''
from tensorflow.keras.callbacks import TensorBoard
import time 
tensorboard_callback = TensorBoard(log_dir=f'./Workbook_ex/IntrotoPython/Ch_16_DeepLearning{time.time()}',histogram_freq=1, write_graph=True)
cnn.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[tensorboard_callback])


print(cnn.summary())

''' Visualize a models structure '''
# from tensorflow.keras.utils import plot_model
# from IPython.display import Image
# plot_model(cnn, to_file='convnet.png', show_shapes=True, show_layer_names=True)
# Image = Image(filename='convnet.png')

''' Evaluate the model '''
loss, accuracy = cnn.evaluate(X_test, y_test)
print(loss, accuracy)

''' Make your predictions (ie. implement the model ) '''
predictions = cnn.predict(X_test)

for index, probability in enumerate(predictions[0]):
    print(f'{index}: {probability:.10%}')

''' Locate and then visualize the incorrect predictions '''
images  = X_test.reshape((10000, 28, 28))
incorrect_predictions = []

for i, (p, e) in enumerate(zip(predictions, y_test)):
    predicted, expected = np.argmax(p), np.argmax(e)

    if predicted != expected:
        incorrect_predictions.append(
            (i, images[i], predicted, expected)
        )

# check how many there are
print(len(incorrect_predictions))

figure, axes = pyplot.subplots(nrows=4, ncols=6, figsize=(16, 12))

for axes, item in zip(axes.ravel(), incorrect_predictions):
    index, image, predicted, expected = item
    axes.imshow(image, cmap=pyplot.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(
        f'index: {index} np: {predicted}; e: {expected}')
pyplot.tight_layout()

''' Display the prediction mistake proabilities '''
def display_probabilities(prediction):
    for index, probability in enumerate(prediction):
        print(f'{index}: {probability:.10%}')

''' Save and load the model '''
cnn.save('mnist_cnn.h5')

# to load do:
# from tensorflow.keras.models import load_model 
# cnn = load_model('mnist_cnn.h5')

