##### Custom Layers #####
import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax
from tensorflow.keras.models import Model

class MyLayer(Layer):
    def __init__(self, units, input_dim):
        super(MyLayer, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal')
        self.b = self.add_weight(shape=(units,), initializer='zeros')

    def call(self, inputs):
        return tf.matmul(inputs, self.w)+self.b

dense_layer = MyLayer(3,5)
x = tf.ones((1,5))
print(dense_layer(x))
print(dense_layer.weights)


# We can also specify if we want to make the Weights and Bias trainable or not.
class MyLayer(Layer):
    def __init__(self, units, input_dim):
        super(MyLayer, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal', trainable=False)
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=False)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)+self.b

dense_layer = MyLayer(3,5)
print('Trainable Weights: ', len(dense_layer.trainable_weights))
print('Non-Trainable Weights: ', len(dense_layer.non_trainable_weights))


### We can even add more functionalities adding variables to customize variables.
class MyLayerMean(Layer):
    def __init__(self, units, input_dim):
        super(MyLayerMean, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal')
        self.b = self.add_weight(shape=(units,), initializer='zeros')
        self.sum_activation = tf.Variable(initial_value=tf.zeros((units,)), trainable=False)
        self.number_call = tf.Variable(initial_value=0, trainable=False)

    def call(self, inputs):
        activations = tf.matmul(inputs, self.w)+self.b
        self.sum_activation.assign_add(tf.reduce_sum(activations, axis=0))
        self.number_call.assign_add(inputs.shape[0])
        return activations, self.sum_activation / tf.cast(self.number_call, tf.float32)

dense_layer = MyLayerMean(3,5)

# See how the values don't change (this is useful to follow the propagation of signals).
y, activation_means = dense_layer(tf.ones((1,5)))
print(activation_means.numpy())
# [ 0.01249229  0.22859478 -0.03365123]

print(activation_means.numpy())
# [ 0.01249229  0.22859478 -0.03365123]


### Now we will create a Dropout layer and add it to a custom Model.
class MyDropout(Layer):
    def __init__(self, rate):
        super(MyDropout, self).__init__()
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate) # here we are using the default Dropout layer defined in TF.

class MyModel(Model):
    def __init__(self, units_1, input_dim_1, units_2, units_3):
        super(MyModel, self).__init__()
        self.layer_1 = MyLayer(units_1, input_dim_1)
        self.dropout_1 = MyDropout(0.5)
        self.layer_2 = MyLayer(units_2, units_1)
        self.dropout_2 = MyDropout(0.5)
        self.layer_3 = MyLayer(units_3, units_2)
        self.softmax = Softmax()

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = tf.nn.relu(x)
        x = self.dropout_2(x)
        x = self.layer_3(x)

        return self.softmax(x)

my_model = MyModel(64, 10000, 64, 46)
print(my_model(tf.ones((1, 10000))))
my_model.summary()