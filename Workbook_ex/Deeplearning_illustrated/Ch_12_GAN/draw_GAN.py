import numpy as np 
import os 
import keras
from keras.models import Model 
from keras.layers import Input, Dense, Conv2D, Dropout
from keras.layers import BatchNormalization, Flatten
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Conv2DTranspose, UpSampling2D
from keras.optimizers import RMSprop

import pandas as pd 
from matplotlib import pyplot as plt 

input_images = 'Workbook_ex/Deeplearning_illustrated/Ch_12_GAN/full_numpy_bitmap_axe.npy'
data = np.load(input_images)

# peek the data 
print(data.shape)

data = data/255
# the division is done to be in the range of 0 and 1 
data = np.reshape(data,(data.shape[0], 28, 28, 1))
img_w, img_h = data.shape[1:3]

# lets check what an example image looks like 
plt.imshow(data[4242,:,:,0], cmap='Greys')
plt.show()

''' Build the discriminator '''

def build_descriminator(depth=64, p=0.4):
    
    # define the inputs
    image = Input((img_w, img_h, 1))
    
    # Conv layers 
    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(image)
    conv1 = Dropout(p)(conv1)
    
    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth*8, 5, strides=2, padding='same', activation='relu')(conv3)
    conv4 = Dropout(p)(conv4)
    
    #output later 
    prediction = Dense(1, activation='sigmoid')(conv4)
    
    # model definition
    model = Model(inputs=image, outputs=prediction)
    
    return model 


discriminator = build_descriminator()

print(discriminator.summary())
discriminator.compile(loss='binary_crossentropy', optimizer= RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0), metrics=['accuracy'])

''' Build the generator '''
z_dimensions = 32

def build_generator(latent_dim= z_dimensions, depth=64, p=0.4):
    
    # define inputs 
    noise = Input((latent_dim,))
    
    # First dense layer 
    dense1 = Dense(7*7*depth)(noise)
    dense1 = BatchNormalization(momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7,7,depth))(dense1)
    dense1 = Dropout(p)(dense1)
    
    # De-Convolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same',activation=None,)(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)
    
    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)
    
    # Output Layer 
    image = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)
    
    # model definition
    model = Model(inputs=noise, outputs=image)
    
    return model 

generator = build_generator()

print(generator.summary())

''' Create the adversarial network '''
z = Input(shape=(z_dimensions,))
img = generator(z)
discriminator.trainable = False 
pred = discriminator(img)
adversarial_model = Model(z, pred)

adversarial_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0004, decay=3e-8, clipvalue=1.0), metrics=['accuracy'])


''' Train the GAN '''

def train( epochs=2000, batch=128, z_dim=z_dimensions):
    
    d_metrics = []
    a_metrics = []
    
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0 
    running_a_acc = 0
    
    for i in range(epochs):
        
        # sample real images 
        real_imgs = np.reshape(
            data[np.random.choice(data.shape[0], batch, replace=False)],
            (batch, 28, 28, 1)
        )
        
        # generate fake images 
        fake_imgs = generator.predict(
            np.random.uniform(-1.0, 1.0, size=[batch, z_dim])
        )
        
        # concate both datasets
        x = np.concatenate((real_imgs, fake_imgs))
        
        # assign the y labels for the descriminator
        y = np.ones([2*batch, 1])
        y[batch:, :] = 0
        
        # train the discriminator
        d_metrics.append(discriminator.train_on_batch(x, y))
        running_d_loss += d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]
        
        # adversarials net noise and real y 
        noise = np.random.uniform(-1.0, 1.0, size=[batch, z_dim])
        y = np.ones([batch, 1])
        
        # train the adversarial net 
        a_metrics.append(
            adversarial_model.train_on_batch(noise, y)
        )
        running_a_loss += a_metrics[-1][0]
        running_a_acc += a_metrics[-1][1]
        
        # periodically print progress and fake images 
        if (i+1)%100 == 0:
            
            print('Epoch #{}'.format(i))
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)
            
            log_mesg = "%d: [A loss: %f, acc: %f]" % (i, running_a_loss/i, running_d_acc/i)
            
            print(log_mesg)
            
            noise = np.random.uniform(-1.0, 1.0, size= [16, z_dim])
            
            gen_imgs = generator.predict(noise)
            
            plt.figure(figsize=(5, 5))
            
            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k+1)
                plt.imshow(gen_imgs[k, :, :, 0],
                           cmap='gray')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
    return a_metrics, d_metrics

# train the gan 
a_metrics_complete, d_metrics_complete = train()

# plotting the results 
ax = pd.DataFrame(
    {
        'Adversarial': [metric[0] for metric in a_metrics_complete],
        'Discriminator': [metric[0] for metric in d_metrics_complete],
    }
).plot(title='Training Loss', logy= True)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
plt.show()

ax = pd.DataFrame(
    {
        'Adversarial': [metric[1] for metric in a_metrics_complete],
        'Discriminator': [metric[1] for metric in d_metrics_complete],
    }
).plot(title='Training Accuracy')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")

plt.show()
