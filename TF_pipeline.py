import tensorflow as tf

# We are going to create different slices of tensors from an array.
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
print (dataset)
for element in dataset:
    print (element)

dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])
print (dataset)
for element in dataset:
    print (element)

# We could also call the numpy array from those tensors.
for element in dataset:
    print (element.numpy())

# We are going to create 128 tensors of length 5.
dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([128,5]))
print (dataset.element_spec)

# More involved example passing a Tuple of tensors to create the dataset.
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([128, 4], minval=1, maxval=10, dtype=tf.int32),
    tf.random.normal([128])))
print (dataset.element_spec)

for elem in dataset.take(2):
    print (elem)

##### Now with an actual dataset ####
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dataset = tf.data.Dataset.from_tensor_slices(((x_train, y_train)))
print(dataset.element_spec)

###### NICE ONE HERE ######
# Using the DataSet class to wrap other generators like ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_datagen = ImageDataGenerator(width_shift_range=0.2, horizontal_flip=True)
dataset = tf.data.Dataset.from_generator(img_datagen.flow, args=[x_train, y_train],
                                         output_types=(tf.float32, tf.int32),
                                         output_shapes=([32,32,32,3], [32,1]))
