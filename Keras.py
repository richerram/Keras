from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy

model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(10,)),
    Dense(1)
])

model.compile (optimizer='sgd', loss=BinaryCrossentropy(from_logits=True))

checkpoint = ModelCheckpoint('my_model.h5', save_weights_only=True)

model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])

############### Then you can LOAD the weights
model.load_weights(('my_model.h5'))

############## Another EXAMPLE to save weights after the model has been trained
###### the same architecture of your model needs to be built.

from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

earlystopping = EarlyStopping(monitor='val_mae', patience=2)

model.fit(X_train, y_train, batch_size=10, validation_split=0.15, callbacks=[earlystopping])

#HERE!!!!!
model.save_weights('my_model.h5')
