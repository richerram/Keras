# Next thing is a very typical thing in TRANSFER LEARNING

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.model import Model

# we can add the parameter "trainable=False" to make the weights of the layer unchangeable.
# If we add them to the layer then the weights will stay as initialized.
inputs = Input(shape=(8,8,1), name='input_layer')
h = Conv2D(16,3, activation='relu', name='conv2d_layer', trainable=False)(inputs)
h = MaxPooling2D(3, name='max_pool2d_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(10, activation='softmax', name='softmax_layer')(h)

model = Model(inputs=inputs, outputs=outputs)

# Another way to do it is to freeze the weights after the model is trained and
# we specify which layer is going to be frozen. We need to do it before it is compiled.
model.get_layer('conv2d_layer').trainable=False

# We can also freeze entire MODELS. In this example we take a model and change the last layer,
# making the last layer the only one that is trainable.
#model = load_model('previous_model')
model.trainable=False
flatten_output = model.get_layer('flatten_layer').output ###### We could've used the input of the outputs layer too.
new_outputs = Dense(5, activation='softmax', name='new_softmax_layer')(flatten_output)
new_model = Model(inputs=inputs, outputs=new_outputs)

new_model.compile(loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)


