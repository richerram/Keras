import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax

model = Sequential(
    [Flatten (input_shape=(28,28)),
     Dense(16, activation = "relu"),
     Dense(16, activation = "relu"),
     Dense(10, activation ="softmax")])
