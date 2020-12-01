import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
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

#print ("train_x: {}\ntest_x: {}\ntrain_y: {}\ntest_y: {}".format(train_x.shape, test_x.shape, train_y.shape, test_y.shape))

'''CALLBACKS Example
class my_callback(Callback):

    def on_train_begin(self, logs=None):
        #Do something at the start of the training

    def on_train_batch_begin(self, batch, logs=None):
        #Do something at the start of every batch iteration

    def on_epoch_end(self, epoch, logs=None):
        #Do something at the end of every Epoch

history = model.fit (train_x, train_y, epochs=100, callbacks = [my_callback()]
'''

def get_regularised_model(wd, rate):
    model = Sequential([
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(wd), input_shape=(train_x.shape[1],)),
        Dropout (rate),
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(wd)),
        Dropout(rate),
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(wd)),
        Dropout(rate),
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(wd)),
        Dropout(rate),
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(wd)),
        Dropout(rate),
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(wd)),
        Dense(1)
    ])
    return model

model=get_regularised_model(1e-2, 0.1)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
hist = model.fit(train_x, train_y, epochs=100, validation_split=0.15, batch_size=64, verbose=False)
for i in hist.history.keys():
    print (i, hist.history[i][-1])
histo2 = model.evaluate(test_x, test_y, verbose=2)
print ("In test sets {} ".format(histo2))

#PLOTTING
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Loss vs Validation Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Training", "Validation"], loc="upper right")
plt.show()

#THIS IS OVERFITTING