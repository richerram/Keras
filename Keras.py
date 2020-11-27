import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, Conv1D

######## Default Weights and Biases #########
# In a "Dense" layer the biases are set to "0" by default  while wheights  are "glorot_uniform"
# Glorot sets weights at random numbers in a closed interval [-c, c], where
# c = sqrt(6 / (n_in + n_out)) where n_in = inputs to, and n_out=outputs from the layer respectively

######## Initializa own Weights and Biases #########
# Weights = kernel_initialiser
# Biases = bias_initialiser
# NOTE : pooling layers don't have weights and biases

model = Sequential ([
    Conv1D (filters = 16, kernel_size=3, input_shape=(128, 64), kernel_initializer="random_uniform", bias_initializer="zeros", activation="relu"),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(64, kernel_initializer="he_uniform", bias_initializer="ones", activation="relu")
])

model.add (Dense(64, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05),
          bias_initializer = tf.keras.initializers.Constant(value = 0.4),
          activation = "relu"))
model.add (Dense (8, kernel_initializer = tf.keras.initializers.Orthogonal(gain = 1.0, seed=None),
           bias_initializer = tf.keras.initializers.Constant(value = 0.4),
           activation = "relu"))

######## Custom Weights and Biases #########
# Must come with two arguments "SHAPE" and "DTYPE"

import tensorflow.keras.backend as K

def my_init (shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64,kernel_initializer=my_init))

fig, axes = plt.subplots(5,2, figsize = (12,16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

weight_layers = [layer for layer in model.layers if len (layer.weights) > 0]

for i, layer in enumerate (weight_layers):
    for j in [0, 1]:
        axes[i, j].hist(layer.weights[j].numpy().flatten(), align='left')
        axes[i, j].set_title(layer.weights[j].name)