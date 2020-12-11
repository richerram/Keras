from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D

inputs = Input(shape=(32,1), name='inputs_layer')
h = Conv1D(16,5, activation='relu', name = 'conv1d_layer')(inputs)
h = AveragePooling1D(3, name='average_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(20, activation='sigmoid', name='outputs_layer')(h)

model = Model(inputs=inputs, outputs=outputs)

# Printing all layers
print(f'-----Model Layers-----\n{model.layers}\n')

#Printing inputs and outputs of a specific leyer
print('-----Inputs of Conv1D layer-----\n{}\n'.format(model.get_layer('conv1d_layer').input))
print('-----Outputs of layer Conv1D-----\n{}\n'.format(model.get_layer('conv1d_layer').output))

# Creating a new model taking the outputs from an Intermediate Layer from the previous model.
flatten_output = model.get_layer('flatten_layer').output
model2 = Model(inputs=model.input, outputs=flatten_output)

#We can crete a new 3rd model using one model, pass it through a Sequential Model and adding Layers
model2 = Sequential([
    model2,
    Dense(10, activation='softmax', name='new_dense_layer')
])
