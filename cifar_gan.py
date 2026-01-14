# GAN: CIFAR-10 dataset

### IMPORTS ###
# Used to create arrays filled with 0s or 1s, for labels
from numpy import zeros, ones

# randn - create random normal numbers, for latent vector for the Generator
# randint - pick a random integer, for choosing random images as Discriminator input
from numpy.random import randn, randint

# Loads the CIFAR-10 dataset, contains 60,000 images in 10 classes
from keras.datasets.cifar10 import load_data

# Optimizer - calculates the internal weights of the model
from keras.optimizers import Adam

# Simple model where layers are stacked linearly
from keras.models import Sequential

# Dense - fully connected layer, every neuron in this layer connects to every neuron in the next
# Reshape - changes the shape of the data, increases dimension of flat list to turn it into an image
# Flatten - opposite of reshape, flattens image into list of numbers for the classifier to read
# Conv2D - convolutional layer, used in Discriminator, scans images to find features
# Conv2DTranspose - transposed convolution, used in Generator, expands noise into a larger image
# LeakyReLu - activation function, allows a small percentage of negative signal to pass through
# Dropout - randomly turns off some neurons during training to prevent overfitting
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout

from matplotlib import pyplot as plt

### LOAD DATASET ###

(trainX, trainy), (testX, testy) = load_data()

# Plot 25 images to see what images are included in this dataset
for i in range(25):
  plt.subplot(5, 5, 1 + i)
  plt.axis('off')
  plt.imshow(trainX[i])
plt.show()

### DISCRIMINATOR ###
# Given an input image, the Discriminator outputs the likelihood of the image being real
# Binary classification - true (1) or false (0) - uses sigmoid activation
def define_discriminator(in_shape=(32, 32, 3)):
  # in_shape defines the input shape, defaults to 32 x 32 pixel image with 3 color channels (RGB)

  model = Sequential()

  # Adds a convolutional layer
  # 128 - filters - model uses 128 different filters
  # (3,3) - kernel_size - each filter is 3x3 pixels big
  # Strides=(2,2) - how many pixels the filter moves at a time as it slides across the image
  #                 (2,2) means it moves 2 pixels at a time to downsample the image
  #                 Downsamples size to 16x16x128 (32/2, 128 filters)
  # padding='same' - adds a buffer (ghost pixels) around the outside of the image
  #                  so the filter can cover edge pixels fully
  model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
  # Alpha determines how many negative signals to pass through
  model.add(LeakyReLU(alpha=0.2))

  # Size 8x8x128 (16/2)
  model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # Flattens 8x8x128 into shape of 8192
  model.add(Flatten())
  # Ignores 40% of the neurons during training, generalize it
  model.add(Dropout(0.4))
  # 1 - outputs a single number (shape of 1)
  # sigmoid - squishes that number in range 0-1
  model.add(Dense(1, activation='sigmoid'))

  # Compile model
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

test_discr = define_discriminator()
print(test_discr.summary())