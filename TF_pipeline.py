# Bidirectional RNNs example #
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, LSTM, Dense, Bidirectional

#Each RNN layer has the attribute "return_sequences" to return an output for each time step.

inputs = Input(shape=(None, 10))            # (None, None, 10)
h = Masking(mask_value=0)(inputs)           # (None, None, 10) # Making sure all 0-padding are ignored
h = LSTM(32, return_sequences=True)(h)      # (None, None, 32)
h = LSTM(64)(h)                             # (None, 64)
outputs = Dense(5, activation='softmax')(h) # (None, 5)

model = Model(inputs, outputs)

# Using the Biderctional layer (wrapper) we create 1 forward layer and 1 backward layer of the type specified,
# duplicating the amount of Features. Notice that this wrapper concatenates de layers by default unless we call
# "merge_mode = sum".

inputs = Input(shape=(None, 10))                        # (None, None, 10)
h = Masking(mask_value=0)(inputs)                       # (None, None, 10)
h = Bidirectional(LSTM(32, return_sequences=True,
                       merge_mode='sum')(h))            # (None, None, 32)
h = Bidirectional(LSTM(32, return_sequences=True))(h)   # (None, None, 64)
h = Bidirectional(LSTM(64))(h)                          # (None, 128)
outputs = Dense(5, activation='softmax')(h)             # (None, 5)