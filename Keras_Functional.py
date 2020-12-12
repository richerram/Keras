import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input

def create_model(layerOne=True, layerTwo=True, layerThree=True):

    model = Sequential([
        Dense(4, input_shape=(4,), activation='relu', kernel_initializer='random_uniform', bias_initializer='ones', trainable=layerOne),
        Dense(2, activation='relu', kernel_initializer='lecun_normal', bias_initializer='ones', trainable=layerTwo),
        Dense(4, activation='softmax', trainable=layerThree)
    ])
    x = len(model.trainable_variables)
    y = len(model.non_trainable_variables)
    print(x)
    print(y)
    return model

model = create_model()

# We retrieve the weights and biases from each layer before training.
def w_layers(model):
    return [e.weights[0].numpy() for e in model.layers]
w0_layers = w_layers(model)

def b_layers(model):
    return [e.bias.numpy() for e in model.layers]
b0_layers = b_layers(model)


# Create a random dataset with train and test sets being equal so we try to learn the identity matrix.
x_train = np.random.random((100,4))
y_train = x_train

x_test = np.random.random((20,4))
y_test = x_test

# Compile and train.
model.compile(optimizer='adam', loss='mse', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, verbose=False)


# Now we retrieve the weights and biases after training.
w0x_layers = w_layers(model)
b0x_layers = b_layers(model)

#Let's plot the variation
def plot_model(w0,b0,w1,b1):
    plt.figure(figsize=(8,8))
    for n in range(3):
        delta_l = w1[n] - w0[n]
        print('Layer '+str(n)+': bias variation: ', np.linalg.norm(b1[n] - b0[n]))
        ax = plt.subplot(1,3,n+1)
        plt.imshow(delta_l)
        plt.title('Layer '+str(n))
        plt.axis('off')
    plt.colorbar()
    plt.suptitle('Weight Matrices Variation')

plot_model(w0_layers, b0_layers, w0x_layers, b0x_layers)

#Now we are going to freeze the first layer and see what happens.
model2 = create_model(layerOne=False)
w1_layers = w_layers(model2)
b1_layers = b_layers(model2)
model2.compile(optimizer='adam', loss='mse', metrics=['acc'])
model2.fit(x_train, y_train, epochs=50, verbose=False)
w1x_layers = w_layers(model2)
b1x_layers = b_layers(model2)
plot_model(w1_layers,b1_layers,w1x_layers,b1x_layers)

#Now we are going to freeze the first and second layers.
model3 = create_model(layerOne=False, layerTwo=False)
w2_layers = w_layers(model3)
b2_layers = b_layers(model3)
model3.compile(optimizer='adam', loss='mse', metrics=['acc'])
model3.fit(x_train, y_train, epochs=50, verbose=False)
w2x_layers = w_layers(model3)
b2x_layers = b_layers(model3)
plot_model(w2_layers,b2_layers,w2x_layers,b2x_layers)