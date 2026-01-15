# GAN: CIFAR-10 dataset
# CIFAR10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

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
# Have to downscale from high-resolution image (32x32) to a single decision
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

### GENERATOR ###
# Given input of latent vector, the Generator produces an image (32x32)
# latent_dim, for example, can be 100, 1D array of size 100
# Given latent_dim (1x100), have to upscale to 32x32x3 image

def define_generator(latent_dim):
  model = Sequential()

  # We reshape the input latent vector into 8x8 image as a starting point, then upscale to 32x32 output
  # 8x8x128 = 8192 nodes
  n_nodes = 8 * 8 * 128
  model.add(Dense(n_nodes, input_dim=latent_dim))
  model.add(LeakyReLU(alpha=0.2))
  # Reshape the list of 8,192 numbers into an image (8x8 pixels wide, 128 layers deep)
  model.add(Reshape((8, 8, 128)))

  # Adds a deconvolutional layer
  # 128 filters, (4,4) kernel size
  # In Conv2DTranspose, strides does the opposite of in Conv2D
  # Instead of shrinking, it doubles (upsamples) the space between pixels and fills those gaps with new data
  # Upsample to 16x16x128
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # Upsample to 32x32x128
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # Generate - output layer
  # Need Conv2D because currently the 32x32 image is 128 layers deep, but a real image only has 3 layers
  # tanh squishes all values to be between -1 and 1
  model.add(Conv2D(3, (8,8), activation='tanh', padding='same'))

  # We don't compile the model because it is not directly trained like the discriminator
  # The Generator is trained via GAN combined model, which will be compiled later

  return model

test_gen = define_generator(100)
print(test_gen.summary())

### GAN - COMBINED MODEL ###
# Define the combined generator and discriminator model, for updating the generator ONLY

def define_gan(generator, discriminator):
  # Set discriminator to not trainable. It is trained separately
  discriminator.trainable = False

  # Connect generator and discriminator
  # Give GAN model latent vector that goes into the Generator, which outputs a new image
  # That image goes into the Discriminator, which outputs real or fake
  model = Sequential()
  model.add(generator)
  model.add(discriminator)

  # Compile model
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)

  return model

### PREPARE TRAINING GAN ###
# We will train the GAN on half-batch of real images and half-batch of fake images

# Load CIFAR Training Images
def load_real_samples():
  (trainX, _), (_, _) = load_data()

  # The pixels are originally stored as integers. Convert to float for math
  X = trainX.astype('float32')

  # Scale from [0, 255] to [-1, 1] to match the output of the Generator
  X = (X - 127.5) / 127.5

  return X

# Pick a half_batch size of random REAL samples
def generate_real_samples(dataset, n_samples):
  # ix - list of random indices
  # dataset.shape[0] - dataset size (60k)
  # n_samples - how many we want to pick
  ix = randint(0, dataset.shape[0], n_samples)

  # X results in an array of the elements from the indices ix
  X = dataset[ix]

  # Generate class labels (answer key) and assign to y
  # Since these are all real images from CIFAR, all 1
  y = ones((n_samples, 1))

  return X, y

# Generates n_samples of latent vectors (size latent_dim) as input for the Generator
def generate_latent_points(latent_dim, n_samples):
  # Generate points in latent space
  x_input = randn(latent_dim * n_samples)

  # Reshape into a batch of inputs for the network
  x_input = x_input.reshape(n_samples, latent_dim)

  return x_input

# Use the Generator to create n fake images, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
  # Generate points in latent space
  x_input = generate_latent_points(latent_dim, n_samples)

  # Use Generator to generate fake examples
  X = generator.predict(x_input)

  # Class labels are 0 as these samples are fake
  y = zeros((n_samples, 1))

  return X, y

### TRAINING ###
# We loop through a number of epochs to train our model
# Train the Discriminator by selecting a random batch of images from the real dataset
# Then, generate a set of fake images with the Generator
# Feed both sets into the Discriminator
# Finally, set the loss parameters for both the real and fake images, and the combined loss

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
  # Batch per epoch
  # = total number of real images / number of images in one batch
  bat_per_epo = int(dataset.shape[0] / n_batch)
  half_batch = int(n_batch / 2)

  # Enumerate epochs
  for i in range(n_epochs):
    # Enumerate batches over the training set
    for j in range(bat_per_epo):

      ### TRAIN DISCRIMINATOR ###
      # The discriminator is trained on half_batch of real images and half_batch fake images, separately
      
      # Randomly select real images
      X_real, y_real = generate_real_samples(dataset, half_batch)
      # Update discriminator model weights, capture loss and ignore accuracy
      d_loss_real, _ = d_model.train_on_batch(X_real, y_real)

      # Generate fake images
      X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      # Update discriminator model weights
      d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)

      ### TRAIN GENERATOR ###

      # Prepare points in latent space as input for the generator
      X_gan = generate_latent_points(latent_dim, n_batch)

      # The generator is trying to trick the discriminator into believing the generated image is true (y = 1)
      y_gan = ones((n_batch, 1))

      # Update the Generator via the Discriminator's error (did G fool D)
      g_loss = gan_model.train_on_batch(X_gan, y_gan)

      print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))

  # Save the generator model
  g_model.save('cifar_generator_model.h5')

### CREATE AND TRAIN MODEL ###
if __name__ == "__main__":
  # Size of latent space
  latent_dim = 100

  # Create the Discriminator
  discriminator = define_discriminator()

  # Create the Generator
  generator = define_generator(latent_dim)

  # Create the GAN
  gan_model = define_gan(generator, discriminator)

  # Load real dataset
  dataset = load_real_samples()

  # Train model
  train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=2)