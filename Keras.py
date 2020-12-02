import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

dataLoad = load_diabetes()

data = dataLoad['data']
targets = dataLoad['target']

train_x, test_x, train_y, test_y = train_test_split(data, targets, test_size=0.1)

model = Sequential([
    Dense(128, activation="relu", input_shape=(train_x.shape[1],)),
    Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1)
])

'''model.compile(optimizer="adam", loss="mse", metrics=["mae"])

class LossAndMetricsCallbacks(Callback):
    def on_train_batch(self, batch, logs=None):
        if batch%2==0:
            print (f"After batch no.{batch} the loss is {logs['loss']}")
    def on_test_batch(self, batch, logs=None):
        print(f"After batch no.{batch} the loss is {logs['loss']}")
    def on_epoch_end(self, epoch, logs=None):
        print (f"Epoch {epoch}: Average loss is {logs['loss']} and MAE is {logs['mae']}")
    def on_predict_batch_end(self, batch, logs=None):
        print (f"Finished prediction on batch: {batch}")

history = model.fit(train_x, train_y, batch_size=100, epochs=20, verbose=False, callbacks=[LossAndMetricsCallbacks()])

evaluation = model.evaluate(test_x, test_y, batch_size=10, verbose=False, callbacks=[LossAndMetricsCallbacks()])

prediction = model.predict(test_x, verbose=False, batch_size=10, callbacks=[LossAndMetricsCallbacks()]'''

################################################################
###################ADVANCED CALLBACK############################
################################################################

lr_schedule = [(4, 0.30), (7, 0.02), (11, 0.005), (15, 0.007)]
def get_new_epoch_lr (epoch, lr):
    epoch_in_sched = [i for i in range(len(lr_schedule)) if lr_schedule[i][0] == int(epoch)]
    if len(epoch_in_sched):
        return lr_schedule[epoch_in_sched[0]][1]
    else:
        return lr


class LRScheduler(tf.keras.callbacks.Callback):

    def __init__(self, new_lr):
        super(LRScheduler, self).__init__()
        # Add the new learning rate function to our callback
        self.new_lr = new_lr

    def on_epoch_begin(self, epoch, logs=None):
        # Make sure that the optimizer we have chosen has a learning rate, and raise an error if not
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Error: Optimizer does not have a learning rate.')

        # Get the current learning rate
        curr_rate = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        # Call the auxillary function to get the scheduled learning rate for the current epoch
        scheduled_rate = self.new_lr(epoch, curr_rate)

        # Set the learning rate to the scheduled learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_rate)
        print('Learning rate for epoch {} is {:7.3f}'.format(epoch, scheduled_rate))

model.compile(loss='mse', optimizer="adam", metrics=['mae', 'mse'])

new_history = model.fit(train_x, train_y, epochs=100,  batch_size=100, validation_split=0.15, verbose=False, callbacks=[EarlyStopping(patience=5)])
# Look how it stops at EPOCh 20-ish

import matplotlib.pyplot as plt

fig = plt.figure (figsize=(5, 12))

fig.add_subplot(211)

plt.plot(new_history.history['loss'])
plt.plot(new_history.history["val_loss"])
plt.title('Loss vs Val_loss')
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.legend(['Training', 'Validation'], loc="upper right")

fig.add_subplot(212)

plt.plot(new_history.history["mae"])
plt.plot(new_history.history['val_mae'])
plt.title('MAE vs Val_MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend(['Training', 'Validation'], loc="upper right")
