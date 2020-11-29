import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

diabetes = load_diabetes()

#print (diabetes["DESCR"])
#print (diabetes.keys())

data = diabetes["data"]
target = diabetes["target"]

target = (target - target.mean(axis=0)) / target.std()
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.1)

print ("train_x: {}\ntest_x: {}\ntrain_y: {}\ntest_y: {}".format(train_x.shape, test_x.shape, train_y.shape, test_y.shape))

def get_model():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(train_x.shape[1],)),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1)
    ])
    return model

model=get_model()
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
hist = model.fit(train_x, train_y, epochs=100, validation_split=0.15, batch_size=64, verbose=False)
model.evaluate(test_x, test_y, verbose=2)

#PLOTTING
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Loss vs Validation Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Training", "Validation"], loc="upper right")
plt.show()

#THIS IS OVERFITTING