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

###############TRAINING CALLBACKS
class TrainingCallback (Callback):
    def on_train_begin (self, logs=None):
        print ('Starting training...')
    def on_epoch_begin (self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
    def on_train_batch_begin (self, batch, logs=None):
        print (f"Training: Starting batch {batch}")
    def on_train_batch_end (self, batch, logs=None):
        print (f"Training: Ending batch {batch}")
    def on_epoch_end(self, epoch, logs=None):
        print(f"Ending epoch {epoch}")
    def on_train_end (self, logs=None):
        print ('Ending training...')

###############TESTING CALLBACKS
class TestingCallback (Callback):
    def on_test_begin (self, logs=None):
        print ('Starting testing...')
    def on_test_batch_begin (self, batch, logs=None):
        print (f"Testing: Starting batch {batch}")
    def on_test_batch_end (self, batch, logs=None):
        print (f"Testing: Ending batch {batch}")
    def on_test_end (self, logs=None):
        print ('Ending testing...')

###############PREDICTION CALLBACKS
class PredictCallback (Callback):
    def on_predict_begin (self, logs=None):
        print ('Starting prediction...')
    def on_predict_batch_begin (self, batch, logs=None):
        print (f"Prediction: Starting batch {batch}")
    def on_predict_batch_end (self, batch, logs=None):
        print (f"Prediction: Ending batch {batch}")
    def on_predict_end (self, logs=None):
        print ('Ending prediction...')

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

model=get_regularised_model(1e-5, 0.3)
model.compile(optimizer="adam", loss="mse")
hist = model.fit(train_x, train_y, epochs=3, validation_split=0.15, batch_size=128, verbose=False, callbacks=[TrainingCallback()])
for i in hist.history.keys():
    print (i, hist.history[i][-1])
histo2 = model.evaluate(test_x, test_y, verbose=2, callbacks=[TestingCallback()])
print ("In test sets {} ".format(histo2))

model.predict(test_x, verbose=2, callbacks=[PredictCallback()])

#PLOTTING
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Loss vs Validation Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Training", "Validation"], loc="upper right")
plt.show()

#THIS IS OVERFITTING