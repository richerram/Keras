#### tf.function Decorator ####
import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt
import time

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
        super(MyModel, self).__init__()
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

model = MyModel(64, 64, 46)

(traindata, trainlabels), (testdata, testlabels) = reuters.load_data(num_words=10000)
class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

word_to_index = reuters.get_word_index()
invert_word_to_index = dict([(value,key) for (key,value) in word_to_index.items()])
text_news = ' '.join([invert_word_to_index.get(i-3, '?') for i in traindata[0]])

### Preprocess the data
def bag_of_words(text_samples, elements=10000):
    output = np.zeros((len(text_samples), elements))
    for i, word in enumerate(text_samples):
        output[i, word] = 1.
    return output
x_train = bag_of_words(traindata)
x_test = bag_of_words(testdata)

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

@tf.function
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


### Print the AUTOGRAPH
print(tf.autograph.to_code(grad.python_function))
'''
def tf__grad(model, inputs, targets, wd):
    do_return = False
    retval_ = ag__.UndefinedReturnValue()
    with ag__.FunctionScope('grad', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        with tf.GradientTape() as tape:
            loss_value = ag__.converted_call(loss, (model, inputs, targets, wd), None, fscope)
        try:
            do_return = True
            retval_ = fscope.mark_return_value((loss_value, ag__.converted_call(tape.gradient, (loss_value, model.trainable_variables), None, fscope)))
        except:
            do_return = False
            raise
    (do_return,)
    return ag__.retval(retval_)'''