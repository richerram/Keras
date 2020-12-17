import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

# Create titles for the columns since they don't exist and import the data.
headers = ['Season', 'Age', 'Disease', 'Trauma', 'Surgery', 'Fever', 'Alcohol', 'Smoking', 'Sitting', 'Output']
fertility = pd.read_csv('fertility_diagnosis.txt', delimiter=',', header=None, names=headers)

# Check shape and verify that the dataframe looks good.
print(fertility.shape)
print(fertility.head)

# Change the value of the output to be Numerical, make all data of typ "float32" and shuffle the data.
fertility['Output'] = fertility['Output'].map(lambda x : 0.0 if x=='N' else 1.0)
fertility = fertility.astype('float32')
fertility = fertility.sample(frac=1).reset_index(drop=True)

# Now we are going to "one-hot-encode" the column SEASON (4 different columns 'cause 4 different values) and move OUTPUT to the end.
fertility = pd.get_dummies(fertility, prefix='Season', columns=['Season'])
fertility.columns = [col for col in fertility.columns if col != 'Output'] + ['Output']
print (fertility.head())

# Convert to NUMPY array.
fertility = fertility.to_numpy()

# Split the data between Training and Test. 70-30
training = fertility[0:70]
test = fertility[70:100]

# Do the x_y split:
x_train = training[:,0:-1]
y_train = training[:,-1]
x_test = test[:,0:-1]
y_test = test[:,-1]

# We create a generator with a batch size of 10.
def get_generator(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield(features[n*batch_size: (n+1)*batch_size], labels[n*batch_size: (n+1)*batch_size])

train_generator = get_generator(x_train, y_train, 10)

next(train_generator)

########################################################
# Moving on to creating the model.
input_shape = (12,)
output_shape =(1,)

inputs = Input(input_shape)
h = BatchNormalization(momentum=0.8)(inputs)
h = Dense(100, activation='relu')(h)
h = BatchNormalization(momentum=0.8)(h)
outputs = Dense(1, activation='sigmoid')(h)

model = Model([inputs], outputs)
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate = 1e-2)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# We will create constants to calculate how many times to run the generator depending on the batch size and how many epochs we will run for.
batch_size = 5
train_steps = len(training) // batch_size
epochs = 3

# Now the loop to run as many generator iterations as available (we will exhaust the generator)

for epoch in range(epochs):
    train_ganerator = get_generator(x_train, y_train, batch_size)
    test_generator = get_generator(x_test, y_test, len(test))
    model.fit_generator(train_generator, steps_per_epoch=train_steps, validation_data=test_generator, validation_steps=1)

############ NOTE you run out of generator data. So we will make it infinite just by adding a While loop, we also add a randomizer of the data.

def get_generator_cyclic(features, labels, batch_size=1):
    while True:
        for n in range(int(len(features)/batch_size)):
            yield(features[n*batch_size: (n+1)*batch_size], labels[n*batch_size: (n+1)*batch_size])
        permuted = np.random.permutation(len(features))
        features = features[permuted]
        labels = labels[permuted]

train_generator_cyclic = get_generator_cyclic(x_train, y_train, batch_size=batch_size)
test_generator_cyclic = get_generator_cyclic(x_test, y_test, batch_size=batch_size)
model.fit_generator(train_generator_cyclic, steps_per_epoch=train_steps, validation_data=test_generator_cyclic, validation_steps=1, epochs=3)

# VALIDATION
validation_generator = get_generator(x_test, y_test, batch_size=30)
predictions = model.predict_generator(validation_generator, steps=1)

# We print the predictions, see how we ROUND the values to get 0 or 1 and then we TRNASPOSE the array to print it in only one line.
print (predictions)
print(np.round(predictions))
print(np.round(predictions.T[0]))

# EVALUATION
validation_generator = get_generator(x_test, y_test, batch_size=30)
print(model.evaluate(validation_generator))
