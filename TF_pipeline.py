##### Custom Training Loops example #####
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD
import numpy as np

my_model = MyModel() #this is a model we have previously created using any known method.

def loss(y_hat, y):
    return tf.reduce_mean(tf.square(y_hat - y))
# the function above is the same as if we used the loss as part of TF library "loss = MeanSquaredError()".

optimizer = SGD(learning_rate=0.05, momentum=0.9)

epoch_losses = []

for epoch in range(num_epochs):
    batch_losses = []

    for inputs, outputs in training_dataset:
        with tf.GradientTape() as tape:
            current_loss = loss(my_model(inputs), outputs)
            grads = tape.gradient(current_loss, my_model.trainable_variables)

        batch_losses.append(current_loss)
        optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

    epoch_losses.append(np.mean(batch_losses))

