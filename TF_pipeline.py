##### Automatic Differentiation #####
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

### We will apply custom differenciation to a simple Linear Regression problem.
def MakeNoisyData(m, b, n=20):
    x = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(len(x),), stddev=0.1)
    y = m * x + b + noise
    return x, y
m = 1
b = 2
x_train, y_train = MakeNoisyData(m,b)
plt.plot(x_train, y_train, 'b.')

class LinearLayer(Layer):
    def __init__(self):
        super(LinearLayer, self).__init__()
        self.m = self.add_weight(shape=(1,), initializer='random_normal')
        self.b = self.add_weight(shape=(1, ), initializer='zeros')
    def call(self, inputs):
        return self.m * inputs + self.b

linear_regression = LinearLayer()
print(linear_regression(x_train))
# tf.Tensor(
# [-1.8976429e-02 -1.7283266e-02 -7.7905636e-03 -2.2711808e-02
#  -1.6863013e-02 -4.8839427e-03 -1.1285791e-02 -1.9753540e-02
#  -8.3114393e-03 -2.7244675e-03 -2.6073331e-02 -1.0096144e-02
#  -4.3933168e-03 -2.1872424e-02 -4.7077634e-03 -1.3091799e-02
#  -2.6780302e-02 -2.2117294e-02 -2.8603491e-03 -5.3068448e-05], shape=(20,), dtype=float32)
print(linear_regression.weights)
# [<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([-0.03286118], dtype=float32)>,
# <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]


### We have to manually define the "Loss Function"
def SquaredError (y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred-y_true))
starting_loss = SquaredError(linear_regression(x_train), y_train)
print('Starting Loss: {}'.format(starting_loss.numpy()))
# Starting Loss: 5.90211820602417


### Training and plotting the model
learning_rate = 0.05
steps = 25

for i in range (steps):
    with tf.GradientTape() as tape:
        predictions = linear_regression(x_train)
        loss = SquaredError(predictions, y_train)
    gradients = tape.gradient(loss, linear_regression.trainable_variables)
    linear_regression.m.assign_sub(learning_rate * gradients[0])
    linear_regression.b.assign_sub(learning_rate * gradients[1])

    print('Step %d, Loss %f' % (i, loss.numpy()))
'''Step 0, Loss 5.902118
Step 1, Loss 4.605634
Step 2, Loss 3.594527
...
Step 24, Loss 0.027175'''

print('m:{}, trained m:{}'.format(m, linear_regression.m.numpy()))
print('b:{}, trained b:{}'.format(b, linear_regression.b.numpy()))
plt.plot(x_train, y_train, 'b.')
x_linear_regression = np.linspace(min(x_train), max(x_train), 50)
plt.plot(x_linear_regression, linear_regression.m * x_linear_regression + linear_regression.b, 'r.')