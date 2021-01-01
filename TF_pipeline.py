##### Flexible Input Shapes #####
import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax
from tensorflow.keras.models import Model

class MyLayer(Layer):
    def __init__(self, units, input_dim, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal')
        self.b = self.add_weight(shape=(units,), initializer='random_normal')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Notice defining the function this way we need to specifiy this values when we instantiete the layer.
dense_layer = MyLayer(3,5)
x = tf.ones((1,5))
print(dense_layer(x))
# tf.Tensor([[0.1397101  0.13544825 0.01942875]], shape=(1, 3), dtype=float32)


### Allowing Flexible Inputs using the "build" method. This is called "lazy" weight creation. The "build" method
# is executed automatically after the "call" method is called.
class MyLayer(Layer):
    def __init__(self, units, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

dense_layer = MyLayer(3)
x = tf.ones((1,5))
print(dense_layer(x))
# tf.Tensor([[-0.20768732  0.13068241 -0.0526246 ]], shape=(1, 3), dtype=float32)

print(dense_layer.weights)
'''[<tf.Variable 'my_layer_2/Variable:0' shape=(5, 3) dtype=float32, numpy=
array([[-0.00211058,  0.04550327, -0.05219404],
       [ 0.01158094,  0.04858255,  0.06237484],
       [-0.05346181,  0.06042183,  0.01163208],
       [-0.05644415,  0.01921833, -0.07422426],
       [-0.10725173, -0.04304357, -0.00021322]], dtype=float32)>, <tf.Variable 'my_layer_2/Variable:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]'''

dense_layer = MyLayer(3)
x = tf.ones((1,1))
print(dense_layer(x))
# tf.Tensor([[ 0.03826848 -0.00380654 -0.04487282]], shape=(1, 3), dtype=float32)

print(dense_layer.weights)
'''[<tf.Variable 'my_layer_8/Variable:0' shape=(1, 3) dtype=float32, numpy=
array([[ 0.03826848, -0.00380654, -0.04487282]], dtype=float32)>, <tf.Variable 'my_layer_8/Variable:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]'''


# We can also benefit from flexible inputs when creating a model and passing outputs from layer to layer.
class MyModel(Model):
    def __init__(self, units1, units2, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.layer1 = MyLayer(units1)
        self.layer2 = MyLayer(units2)
    def call(self, inputs):
        x = self.layer1(inputs)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        return Softmax()(x)

model = MyModel(units1=32, units2=10)
_ = model(tf.ones((1,100))) #to create and initialize all the weights
model.summary()

