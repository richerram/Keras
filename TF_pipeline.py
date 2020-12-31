##### Custom Layers example #####
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model

class LinearMap(Layer):
    def __init__(self, input_dim, units):
        super(LinearMap, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units)))

        # An analog way for self.w could be this:
        # self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal') #

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

linear_layer = LinearMap(3,2)

input = tf.ones((1,3))
print(linear_layer(input))
# tf.Tensor([[-0.09535347 -0.16383061]], shape=(1, 2), dtype=float32)

print(linear_layer.weights)
'''[<tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.06317208, -0.0368916 ],
       [-0.06696137, -0.05961837],
       [ 0.03477998, -0.06732064]], dtype=float32)>]'''


### Now we create a Model using our custom Layer

class MyModel(Model):
    def __init__(self, hidden_units, outputs, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = Dense(hidden_units, activation='sigmoid')
        self.linear = LinearMap(hidden_units, outputs)

    def call(self, inputs):
        h = self.dense(inputs)
        return self.linear(h)

my_model = MyModel(64, 12, name='my_custom_model')