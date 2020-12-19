# Training with Datasets - Full Example #
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization

bank_pd = pd.read_csv('bank-full.csv', delimiter=';')

# Pandas options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


bank_pd.head()
print (bank_pd.shape)

# Let's ONE-HOT ENCODE with sklearn.
features = ['age','job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'campaign', 'pdays', 'poutcome']
labels = ['y']
bank_pd = bank_pd.filter(features + labels)

encoder = LabelBinarizer()
categorical_features = ['default', 'housing', 'job', 'loan', 'education', 'contact', 'poutcome']
for feature in categorical_features:
    bank_pd[feature] = tuple(encoder.fit_transform(bank_pd[feature]))

print(bank_pd.head())

# Shuffle the dataframe
bank_pd = bank_pd.sample(frac=1).reset_index(drop=True)
print(bank_pd.head())

# We are going to make the dataset through a dictionary
bank_dataset = tf.data.Dataset.from_tensor_slices(bank_pd.to_dict(orient='list'))
bank_dataset.element_spec

# Filtering the dataset to get rid of all non-divorced people.
def check_divorced():
    bank_dataset_iter = iter(bank_dataset)
    for i in bank_dataset_iter:
        if i['marital'] != 'divorced':
            print('Found a person with marital status {}'.format(i['marital']))
            return
    print('No non-divorced people were found')
check_divorced()

bank_dataset = bank_dataset.filter(lambda x: tf.equal(x['marital'], tf.constant([b'divorced']))[0])
check_divorced()


# Mapping a function to make the label "y" categorical 0 or 1.
def map_label(x):
    x['y'] = 0 if (x['y'] == tf.constant([b'no'], dtype=tf.string)) else 1
    return x
bank_dataset = bank_dataset.map(map_label)
bank_dataset.element_spec

# Now we will filter out the Marital column since it is the only string.
bank_dataset = bank_dataset.map(lambda x: {key:val for key,val in x.items() if key != 'marital'})
bank_dataset.element_spec
print(bank_dataset.take(1))

# Create an (input, output) Tuple for the dataset.
def map_feature(x):
    features = [[x['age']], [x['balance']], [x['campaign']], x['contact'], x['default'], x['education'],
                x['housing'], x['job'], x['loan'], [x['pdays']], x['poutcome']]
    return (tf.concat(features, axis=0), x['y'])
bank_dataset = bank_dataset.map(map_feature)
bank_dataset.element_spec

# Now split between training and test sets, first we need to count.
data_len = 0
for i in bank_dataset:
    data_len += 1
print (data_len

train_num = int(data_len * 0.7)
train_set = bank_dataset.take(train_num)
test_set = bank_dataset.skip(train_num)

# Build the model
model = Sequential()
model.add(Input(shape=(30,)))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
model.summary()

# Setting up and training
train_set = train_set.batch(20, drop_remainder=True)
test_set = test_set.batch(100)

train_set = train_set.shuffle(1000)

plt.plot (history.epoch, history.history['acc'], label='training')
plt.plot (history.epoch, history.history['val_acc'], label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

history = model.fit(train_set, validation_data=test_set, epochs=5)