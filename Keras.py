import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential ([
    Conv2D (16, (3,3), padding = "SAME", strides = 2, activation = "relu", input_shape= (32, 32, 3)),
    MaxPooling2D ((3,3)),
    Flatten (),
    Dense (16, activation = "relu"),
    Dense (10, activation = "softmax")
])