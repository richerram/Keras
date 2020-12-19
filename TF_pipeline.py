##### Training with Datasets examples #####
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
print (dataset.element_spec)

# Now, something we can do is to train the data in BATCHES. Notice how when we print the NONE is added.
dataset = dataset.batch(16)
print (dataset.element_spec)

# None is shown since we dont know how big will the last batch be, but we can get rid of that.
dataset = dataset.batch(16, drop_remider=True)
print (dataset.element_spec)

# To train it we only need to call the dataset, no need to add inputs and outputs.
history = model.fit(dataset) # This will only train for 0ne epoch or 1 pass through all the elements in the dataset object.

# To add epochs.
dataset = dataset.repeat(10)
# or this way:
dataset = dataset.repeat() #repeats indefinitely
history = model.fit(dataset, steps_per_epoch=x_train.shape[0]//16, epochs=10)

# We can also randomly shuffle the dataset before training.
dataet = dataset.shuffle(100) #buffer size of 100 and from this 16 we take 16 on every batch.

# We can add transformations to our dataset before training using the MAP and FILTER functions.
def rescale(image, label):
    return image/255, label
dataset = dataset.map(rescale)

def label_filter(image, label):
    return tf.squeeze(label) != 9
dataset = dataset.filter(label_filter())

dataset = dataset.shuffle(100)
dataset = dataset.batch(16, drop_reminder=True)
dataet = dataset.repeat()

history = model.fit(dataset, steps_per_epoch=x_train[0]//16, epochs=10)