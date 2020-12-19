import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D
import os

x = np.zeros((100,10,2,2))

dataset1=tf.data.Dataset.from_tensor_slices(x)
print(dataset1)
print(dataset1.element_spec)

# If we try to create a dataset with elements of different sizes it creates an error.
x2 = [np.zeros((10,2,2)), np.zeros((5,2,2))]
# dataset2=tf.data.Dataset.from_tensor_slices(x2) # This creates an error.

x2 = [np.zeros((10,1)), np.zeros((10,1)), np.zeros((10,1))]
dataset2 = tf.data.Dataset.from_tensor_slices(x2)

# Using "zipping" to combine datasets. Sizes (shapes) of the datasets don't need to be the same but the final
# dataset will truncate to the smaller dataset' shape.
zipped_dataset = tf.data.Dataset.zip((dataset1, dataset2))
print(zipped_dataset.element_spec)

# We create a "get_batches" function to see how many batches were created.
def get_batches(dataset):
    iter_dataset = iter(dataset)
    i = 0
    try:
        while next(iter_dataset):
            i = i+1
    except:
        return i

get_batches(zipped_dataset)

######## Datasets from Numpy Arrays #########
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(type(x_train), type(y_train))

mnist_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
print(mnist_dataset)

# Inspecting the insides
element = next(iter(mnist_dataset.take(1)))
print (len(element))
print(element[0].shape)
print(element[1].shape)

# Another example with text files.
text_files = sorted([f.path for f in os.scandir('shakespeare')])
print(text_files)

with open(text_files[0], 'r') as fil:
    contents = [fil.readline() for i in range(5)]
    for line in contents:
        print(line)

shak_dataset = tf.data.TextLineDataset(text_files)

#To find the insides.
first5lines = iter(shak_dataset.take(5))
lines = [line for line in first5lines]
for line in lines:
    print (line)

# We can see that all the files are in our dataset:
shak_dataset_iterator = iter(shak_dataset)
lines = [line for line in shak_dataset_iterator]
print (len(lines))

# If we'd like to print the first lines of each file we would have to jump from one file to another before running through all the lines.
# For that we would use the INTERLEAVE function.

text_file_dataset = tf.data.Dataset.from_tensor_slices(text_files)
files = [file for file in text_file_dataset]
for file in files:
    print (file)

interleaved_shak = text_file_dataset.interleave(tf.data.TextLineDataset, cycle_length=9)
print (interleaved_shak)

lines = [line for line in iter(interleaved_shak.take(10))]
for line in lines:
    print(line)