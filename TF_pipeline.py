#### Model Subclassing example #####
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Note the subclassing class uses the Model class.
# Basic example
class MyModel(Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = Dense(16)

    def call(self, inputs):
        return self.dense(inputs)
my_model = MyModel(name='my_model')


# Basic example #2
class MyModel(Model):
    def __init__(self, num_classes, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense1 = Dense(16, activation='sigmoid')
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        h = self.dense1(inputs)
        return self.dense2(h)
my_model = MyModel(10, name='my_model')

# Basic example #3
# Notice here we use the "training=False" argument, very useful with Dropout and BatchNorm layers.
# This tells the model to use those layers when testing.
class MyModel(Model):
    def __init__(self, num_classes, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense1 = Dense(16, activation='sigmoid')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        h = self.dense1(inputs)
        h = self.dropout(h, training=training)
        return self.dense2(h)
my_model = MyModel(12, name='my_model')