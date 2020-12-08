import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, layers

# Loading data
pd_dat = pd.read_csv('diagnosis.csv')
dataset = pd_dat.values

#Splitting data
train_x, test_x, train_y, test_y = train_test_split(dataset[:,:6], dataset[:,6:], test_size=0.33)

# Creating individual input and objective matrices
temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train = np.transpose(train_x)
X_train = [temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train]
temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test = np.transpose(test_x)
X_test = [temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test]

inflam_train, nephr_train = train_y[:,0], train_y[:,1]
y_train = [inflam_train, nephr_train]
inflam_test, nephr_test = test_y[:,0], test_y[:,1]
y_test = [inflam_test, nephr_test]

############## Creating model #################
# First we create INPUT layers to then concatenate to 'x' so they can be fed into the 2nd layer (DENSE) as only 1 input (x).
shape_inputs = (1,)
temperature = Input(shape=shape_inputs, name='temp')
nausea_occurence = Input(shape=shape_inputs, name='nocc')
lumbar_pain = Input(shape=shape_inputs, name='lumbp')
urine_pushing = Input(shape=shape_inputs, name='up')
micturition_pain = Input(shape=shape_inputs, name='mict')
bis = Input(shape=shape_inputs, name='bis')

# Concatenation and then split and merge again.
list_inputs = [temperature, nausea_occurence, lumbar_pain, urine_pushing, micturition_pain, bis]
x = layers.concatenate(list_inputs)
xa = layers.Dense(1, activation='relu')(x)
xb = layers.Dense(1, activation='sigmoid')(x)
xc = layers.concatenate([xa,xb])

# Creation of the 2 ""model classifiers/outputs"", one for each disease
inflamation_prediction = layers.Dense(1, activation='sigmoid', name='inflam')(xc)
nephritis_prediction = layers.Dense(1, activation='sigmoid', name='nephr')(xc)
list_outputs = [inflamation_prediction, nephritis_prediction]

# Creating MODEL object
model = tf.keras.Model(inputs=list_inputs, outputs=list_outputs)

# Plotting the model
tf.keras.utils.plot_model(model, 'multi_input_output_model.png', show_shapes=True)

# Compiling the model.
model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
              loss={'inflam':'binary_crossentropy', 'nephr':'binary_crossentropy'},
              metrics={'inflam':['acc'], 'nephr':['acc']},
              loss_weights=[1,0.2])
#Fit test
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, verbose=False)

#PLOT
acc_keys = [k for k in history.history.keys() if k in ('inflam_acc', 'nephr_acc')]
loss_keys = [k for k in history.history.keys() if k not in acc_keys]

for k, v in history.history.items():
    if k in acc_keys:
        plt.figure(1)
        plt.plot(v)
    else:
        plt.figure(2)
        plt.plot(v)

plt.figure(1)
plt.title('Accuracy vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(acc_keys)

plt.figure(2)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loss_keys)

#Evaluate model with the Test Set
model.evaluate(X_test, y_test, verbose=2)
