from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D
from tensorflow.keras.models import Model

# Example using LISTS as parameters, all parameters are assigned in the same order as they were created in the MODEL.

inputs = Input(shape=(32,1))
h = Conv1D(16,5, activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = FLatten()(h)
aux_inputs = Input(shape=(12,))
h = Concatenate()([h, aux_inputs])
outputs =  Dense(20, activation='sigmoid')(h)
aux_outputs = Dense(1, activation='linear')(h)

model = Model(inputs=[inputs, aux_inputs], outputs=[outputs, aux_outputs])
mode.compile (loss=['binary_crossentropy', 'mse'], loss_weights=[1,0.4], metrics=['accuracy'])
history = model.fit([X_train, X_aux], [y_train, y_aux], validation_split=0.2, epochs=20)

##########################################################################################
# Example using DICTIONARIES

inputs = Input(shape=(32,1), name='inputs')
h = Conv1D(16,5, activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = FLatten()(h)
aux_inputs = Input(shape=(12,), name='aux_inputs')
h = Concatenate()([h, aux_inputs])
outputs =  Dense(20, activation='sigmoid', name='outputs')(h)
aux_outputs = Dense(1, activation='linear', name='aux_outputs')(h)

model = Model(inputs=[inputs, aux_inputs], outputs=[outputs, aux_outputs])
model.compile(loss={'outputs':'binary_crossentropy', 'aux_outputs':'mse'}, loss_weights={'outputs':1, 'aux_outputs':0.4}, metrics=['accuracy'])
history = model.fit({'inputs':X_train, 'aux_inputs':X_aux},
                    {'outputs':y_train, 'aux_outputs':y_aux},
                    epochs=20, validation_split=20)
