from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from training import generate_latent_points

# Plot generated images
def show_plot(examples, n):
  for i in range(n * n):
    plt.subplot(n, n, 1 + i)
    plt.axis('off')
    plt.imshow(examples[i, :, :, :])
  plt.show()

# Load model
model = load_model('cifar_generator_model.h5')

# Generate images
latent_points = generate_latent_points(100, 25)
X = model.predict(latent_points)

# Scale from [-1, 1] to [0, 1]
X = (X + 1) / 2.0

X = (X*255).astype(np.uint8)

# Plot the result
show_plot(X, 5)