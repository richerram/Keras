import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

train_x = train_x [:10000]
train_y = train_y [:10000]
test_x = test_x [:1000]
test_y = test_y [:1000]

import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 10, figsize=(10,1))
# for i in range(10):
#     ax[i].set_axis_off()
#     ax[i].imshow(train_x[i])

def get_test_accuracy (model, x, y):
    test_loss, test_accuracy = model.evaluate (x, y, verbose=0)
    print (f'accuracy {test_accuracy}')

def get_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32,32,3), kernel_size=(3,3), activation='relu', name='conv_1'),
        Conv2D(filters=8, activation='relu', kernel_size=(3,3), name='conv_2'),
        MaxPooling2D(pool_size=(4,4), name='pool_1'),
        Flatten(name='flatten_1'),
        Dense(32, activation='relu', name='dense_1'),
        Dense(units=10, activation='softmax', name='dense_2' )
    ])

    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = get_model()
print (get_test_accuracy(model, test_x, test_y))

############## CHECKPOINT #######################
#################################################
checkpoint_path = 'model_checkpoints/checkpoint'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, frequency='epoch', save_weights_only=True, verbose=False)

#odel.fit (train_x, train_y, epochs=100, callbacks=[checkpoint])

model.load_weights(checkpoint_path)
print(get_test_accuracy(model, test_x, test_y))

######################Clear Directory (on Lixux command)########################
# ! rm -r model_checkpoints