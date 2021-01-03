##### Custom Training Loops #####
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Softmax
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

### We will create the custom layers (accepting variable input shape) and the custom model.
class MyLayer(Layer):
    def __init__(self, units):
        super(MyLayer, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', name='kernel')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class MyDropout(Layer):
    def __init__(self, rate):
        super(MyDropout, self).__init__()
        self.rate = rate
    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)

class MyModel(Model):
    def __init__(self, units1, units2, units3):
        super (MyModel, self).__init__()
        self.layer1 = MyLayer(units1)
        self.dropout1 = MyDropout(0.5)
        self.layer2 = MyLayer(units2)
        self.dropout2 = MyDropout(0.5)
        self.layer3 = MyLayer(units3)
        self.softmax = Softmax()
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        return self.softmax(x)

### Now we instanciate a model with the inputs according to the Reuters dataset.
model = MyModel(64, 64, 46)
print(model(tf.ones((1,10000))))
model.summary()

### Now we load the Reuters dataset
(traindata, trainlabels), (testdata, testlabels) = reuters.load_data(num_words=10000)
class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

print('Label: {}'.format(class_names[trainlabels[0]]))
# earn

word_to_index = reuters.get_word_index()
invert_word_to_index = dict([(value,key) for (key,value) in word_to_index.items()])
text_news = ' '.join([invert_word_to_index.get(i-3, '?') for i in traindata[0]])
print(text_news)
'''? ? ? said as a result of its december acquisition of space co it expects earnings per share in 1987 
of 1 15 to 1 30 dlrs per share up from 70 cts in 1986 the company said pretax net should rise to nine 
to 10 mln dlrs from six mln dlrs in 1986 and rental operation revenues to 19 to 22 mln dlrs from 12 5 mln 
dlrs it said cash flow per share this year should be 2 50 to three dlrs reuter 3'''


### Preprocess the data
def bag_of_words(text_samples, elements=10000):
    output = np.zeros((len(text_samples), elements))
    for i, word in enumerate(text_samples):
        output[i, word] = 1.
    return output
x_train = bag_of_words(traindata)
x_test = bag_of_words(testdata)

print('Shape of x_train: ', x_train.shape)
# Shape of x_train:  (8982, 10000)
print('Shape of x_test: ', x_test.shape)
# Shape of x_test:  (2246, 10000)


### Define the "loss function" and compute "backward" and "forward" pass.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

def loss(model, x, y, wd):
    kernel_variables = []
    for l in model.layers:
        for w in l.weights:
            if 'kernel' in w.name:
                kernel_variables.append(w)
    wd_penalty = wd * tf.reduce_sum([tf.reduce_sum(tf.square(k)) for k in kernel_variables])
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_) + wd_penalty

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


### Train the model
def grad(model, inputs, targets, wd):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, wd)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

start_time = time.time()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, trainlabels))
train_dataset = train_dataset.batch(32)

train_loss_results = [] # keep results for plotting
train_accuracy_results = []

num_epochs = 10
weight_decay = 0.005

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for x, y in train_dataset: # Training Loop
        loss_value, grads = grad(model, x, y, weight_decay) # Optimize the model
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg(loss_value) # Compute current loss
        epoch_accuracy(to_categorical(y), model(x)) # Compare predicted label to actual label

    train_loss_results.append(epoch_loss_avg.result()) # End Epoch
    train_accuracy_results.append(epoch_accuracy.result())
    print('Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

print('Duration :{:.3f}'.format(time.time() - start_time))
'''Epoch 000: Loss: 1.797, Accuracy: 67.836%
Epoch 001: Loss: 1.701, Accuracy: 71.365%
Epoch 002: Loss: 1.657, Accuracy: 73.469%
Epoch 003: Loss: 1.638, Accuracy: 74.415%
Epoch 004: Loss: 1.636, Accuracy: 75.529%
Epoch 005: Loss: 1.608, Accuracy: 76.330%
Epoch 006: Loss: 1.611, Accuracy: 76.531%
Epoch 007: Loss: 1.601, Accuracy: 77.054%
Epoch 008: Loss: 1.597, Accuracy: 77.310%
Epoch 009: Loss: 1.585, Accuracy: 77.678%
Duration :46.206'''


### Evaluate Model
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, testlabels))
test_dataset = test_dataset.batch(32)

epoch_loss_avg = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

for x, y in test_dataset:
    loss_value = loss(model, x, y, weight_decay)
    epoch_loss_avg(loss_value)
    epoch_accuracy(to_categorical(y), model(x))

print('Test Loss: {:.3f}'.format(epoch_loss_avg.result().numpy()))
# Test Loss: 1.786
print('Test Accuracy: {:.3%}'.format(epoch_accuracy.result().numpy()))
# Test Accuracy: 71.594%


### Plots
fig, axes = plt.subplots(2, sharex=True, figsize=(12,8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel('Loss', fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


### Predict from label
predic_label = np.argmax(model(x_train[np.newaxis, 0]), axis=1)[0]
print('Prediction: {}'.format(class_names[predic_label]))
# Prediction: earn
print('     Label: {}'.format(class_names[trainlabels[0]]))
#      Label: earn

