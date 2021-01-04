#### Tracking metrics in custom training loops #####
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GRU, Bidirectional, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Mean, AUC
from tensorflow import GradientTape
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000, skip_top=50)
class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

padded_train_data = pad_sequences(train_data, maxlen=100, truncating='post')
padded_test_data = pad_sequences(test_data, maxlen=100, truncating='post')

train_data, val_data, train_labels, val_labels = train_test_split(padded_train_data, train_labels, test_size=0.3)

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((padded_test_data, test_labels))
test_dataset = test_dataset.batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.shuffle(500)
val_dataset = val_dataset.batch(32)

class RNNModel (Model):
    def __init__(self, units_1, units_2, num_classes, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.embedding = Embedding(input_dim=10000, output_dim=16, input_length=100)
        self.gru_1 = Bidirectional(GRU(units_1, return_sequences=True), merge_mode='sum')
        self.gru_2 = GRU(units_2)
        self.dense = Dense(num_classes, activation='softmax')
    def call(self, inputs):
        h = self.embedding(inputs)
        h = self.gru_1(h)
        h = self.gru_2(h)
        return self.dense(h)

model = RNNModel(units_1=32, units_2=16, num_classes=46, name='rnn_model')
optimizer = SGD(learning_rate=0.005, nesterov=True)
loss = SparseCategoricalCrossentropy()


### Custom Training Loop
def grad(model, inputs, targets, loss):
    with GradientTape() as tape:
        preds = model(inputs)
        loss_value = loss(targets, preds)
    return preds, loss_value, tape.gradient(loss_value, model.trainable_variables)

# Metric objects can be created and used to track performance measures in the custom training loop. We will set up our
# custom training loop to track the average loss, and area under the ROC curve (ROC AUC). Of course there are many more
# metrics that you could use.
train_loss_results = []
train_roc_auc_results = []
val_loss_results = []
val_roc_auc_results = []
# In the following custom training loop, we define an outer loop for the epochs, and an inner loop for the batches
# in the training dataset. At the end of each epoch we run a validation loop for a number of iterations.
#
# Inside the inner loop we use the metric objects to calculate the metric evaluation values.
# These values are then appended to the empty lists. The metric objects are re-initialised at the start of each epoch.
num_epochs = 5
val_steps = 10

for epoch in range(num_epochs):
    train_epoch_loss_avg = Mean()
    train_epoch_roc_auc = AUC(curve='ROC')
    val_epoch_loss_avg = Mean()
    val_epoch_roc_auc = AUC(curve='ROC')

    for inputs, labels in train_dataset:
        model_preds, loss_value, grads = grad(model, inputs, labels, loss)
        optimizer.apply_gradients((zip(grads, model.trainable_variables)))

        train_epoch_loss_avg(loss_value)
        train_epoch_roc_auc(to_categorical(labels, num_classes=46), model_preds)

    for inputs, labels in val_dataset.take(val_steps):
        model_preds = model(inputs)
        val_epoch_loss_avg(loss(labels, model_preds))
        val_epoch_roc_auc(to_categorical(labels, num_classes=46), model_preds)

    train_loss_results.append(val_epoch_loss_avg.result().numpy())
    train_roc_auc_results.append(val_epoch_roc_auc.result().numpy())
    val_loss_results.append(val_epoch_loss_avg.result().numpy())
    val_roc_auc_results.append(val_epoch_roc_auc.result().numpy())

    print("Epoch {:03d}: Training loss: {:.3f}, ROC AUC: {:.3%}".format(epoch, train_epoch_loss_avg.result(),
                                                                        train_epoch_roc_auc.result()))
    print("              Validation loss: {:.3f}, ROC AUC {:.3%}".format(val_epoch_loss_avg.result(),
                                                                         val_epoch_roc_auc.result()))

# Plots
fig = plt.figure(figsize=(15,5))

fig.add_subplot(121)
plt.plot(train_loss_results)
plt.plot(val_loss_results)
plt.title('Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='best')

fig.add_subplot(122)
plt.plot(train_roc_auc_results)
plt.plot(val_roc_auc_results)
plt.title('ROC Accuracy vs Epoch')
plt.ylabel('ROC Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='best')
plt.show()


### Testing the model
test_epoch_loss_avg = Mean()
test_epoch_roc_auc = AUC(curve='ROC')

for inputs, labels in test_dataset:
    model_preds = model(inputs)
    test_epoch_loss_avg(loss(labels, model_preds))
    test_epoch_roc_auc(to_categorical(labels, num_classes=46), model_preds)

print("Test loss: {:.3f}".format(test_epoch_loss_avg.result().numpy()))
# Test loss: 2.525
print("Test ROC AUC: {:.3%}".format(test_epoch_roc_auc.result().numpy()))
# Test ROC AUC: 85.936%