import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10

# Import cifar10 data and split it.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# We are converting the labels "to categorical" for easier management.
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Create a generator function (same as previous example), then call it to create a generator.
def get_gen(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield(features[n*batch_size:(n+1)*batch_size], labels[n*batch_size:(n+1)*batch_size])
training_gen = get_gen(x_train, y_train)

# We call "next" to get an image and then we plot it.
image, label = next(training_gen)
image_unbatched = image[0,:,:,:]
plt.imshow(image_unbatched)
print(label)

#Let's reset the generator.
training_gen = get_gen(x_train, y_train, batch_size=512)

#####     DATA AUGMENTATION GENERATOR     #####
# We are going to create a function to process an image during the process of calling the image.

def monochrome(x):
    def func_bw(a):
        avrg_color = np.mean(a)
        return [avrg_color, avrg_color, avrg_color]
    x = np.apply_along_axis(func_bw, -1, x)
    return x

# Data Generator Object
img_gen = ImageDataGenerator(preprocessing_function=monochrome, rotation_range=180, rescale=(1/255.0))
img_gen.fit(x_train)

# # Now we do create the generator using "flow".
# img_gen_iterable = img_gen.flow(x_train, y_train, batch_size=512, shuffle=False)
#
# # We print an image from the transformation and compare it to the original.
# image, label = next(img_gen_iterable)
# image_orig, label_orig = next(training_gen)
# figs, axes = plt.subplots(1,2)
# axes[0].imshow(image[0,:,:,:])
# axes[0].set_title('Transformed')
# axes[1].imshow(image_orig[0,:,:,:])
# axes[1].set_title('Original')
# plt.show()

# We create the model.
model = Sequential()
model.add(Input((32,32,3)))
model.add(Conv2D(8, (32,32), padding='same', activation='relu'))
model.add(MaxPooling2D((4,4)))
model.add(Conv2D(8, (8,8), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(4, (4,4), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Train the model calculating steps per ecpoch for training and testing.
# train_steps_per_epoch = training_gen.n // training_gen.batch_size
# print (train_steps_per_epoch)

model.fit_generator(training_gen, steps_per_epoch=512, epochs=5)