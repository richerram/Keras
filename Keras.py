import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


model = Sequential ([Conv2D(16,(3,3), activation="relu", input_shape=(28,28,1)),
                    MaxPooling2D((3,3)),
                    Flatten(),
                    Dense(10,activation="softmax")])

opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=[acc, mae])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

labels = ["t-shirts", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]

x_train = x_train / 255
x_train = x_train[...,np.newaxis]
x_test = x_test / 255
x_test = x_test[...,np.newaxis]

# i = 43
# img = x_train[i, :, :]
# plt.imshow(img)
# plt.show()
# print (f"label: {labels[y_train[i]]}")

#history = model.fit(x_train, y_train, epochs=10, verbose=2)
df = pd.DataFrame(history.history)

loss_plot = df.plot(y="loss")

model.evaluate(x_test, y_test)
model.predict(x_test[49])