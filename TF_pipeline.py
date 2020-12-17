# Dataset Generators Example
# This is a good way of feeding data into our model that doesn't fit in our Memory.

# Basic YIELD function
def text_file_reader(filepath):
    with open(filepa, 'r') as f:
        for row in f
            yield row
text_data_generator = text_file_reader('path to file.txt')

next(text_data_generator) # Get one line at a time.
next(text_data_generator)


# Generating an infinte series of values.
import numpy as np

def get_data(batch_size):
    while True:
        y_train = np.random.choice([0,1], (batch_size, 1))
        x_train = np.random.randn(batch_size, 1) + (2 * y_train - 1)
        yield x_train, y_train

datagen = get_data(32)
x, y = next(datagen)

# Now we create the Tensorflow Model to feed this data.
# Logistic Regression Model with Gradient Decent optimizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(1, activation='sigmoid')])
model.compile(optimizer='sgd', loss='binary_crossentropy')

# Now we use de method ".fit_generator" where we pass the generator object and
# since we know how many iterations is an epoch we use the "steps_per_epoch" argument.

model.fit_generator(datagen, steps_per_epoch=1000, epochs=10)

'''As a side (not too recommended) option we can use the "train_on_batch" method,
this can be used only when there is the need to do some pre-process to the data we are
taking from the generator
gi 
for _ in range (10000):
    x_train, y_train = next(datagen)
    model.train_on_batch(x_train, y_train)
'''

# Also, we could've created enetators for the evaluation and prediction functions and specify how many steps to run.
model.evaluate_generator(datagen_eval, steps=100)
model.predict_generator(datagen_test, steps=100)