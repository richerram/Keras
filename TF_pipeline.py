##### tf.function Decorator example #####
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD

my_model = MyModel()
loss = MeanSquaredError()
optimizer = SGD(learning_rate=0.05, momentum=0.9)

@tf.function ################################ this decorator turns it into a graph.
def get_loss_and_grads(inputs, outputs):
    with tf.GradientTape() as tape:
        current_loss = loss(my_model(inputs), outputs)
        grads = tape.gradient(current_loss, my_model.trainable_variables)
    return current_loss, grads

for epoch in range(num_epochs):
    for inputs, outuputs in training_dataset:
        curren_loss, grads = get_loss_and_grads(inputs, outputs)
        optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

