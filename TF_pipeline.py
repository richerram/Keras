##### Padding and Masking the IMDB dataset #####

import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data()
print(type(x_train))
# <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (25000,) (25000,) (25000,) (25000,)

# Let's display the first element of the dataset.
print(x_train[0])
''' [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 
    4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 
    838, 112, 50, 670, 22665, 9, 35, 480, 284, 5, 150,...'''
print(y_train[0])
# 1 ---> this means it is a positive review

# Import the actual words represented by those indexes.
jsonwords = imdb.get_word_index()
# If we want to map the words to the indices we need to consider that the "index_from" starts from 3 in the dataset,
# why this is not mentioned anywhere in the documentation? I don't know!!!
index_from = 3
jsonwords = {k:v + index_from for k,v in jsonwords.items()}

# Let's print the sentence.
print(*[k for i in x_train[0] for k,v in jsonwords.items() if v==i])
'''
this film was just brilliant casting location scenery story direction everyone's really suited the part they played and 
you could just imagine being there robert redford's is an amazing actor and now the same being director norman's father
came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty 
remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released 
for retail and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was 
so sad and you know what they say if you cry at a film it must have been good and this definitely was also congratulations 
to the two little boy's that played the part's of norman and paul they were just brilliant children are often left out 
of the praising list i think because the stars that play them all grown up are such a big profile for the whole film but 
these children are amazing and should be praised for what they have done don't you think the whole story was so lovely 
because it was true and was someone's life after all that was shared with us all
'''

print('GPU name: {}', format(tf.test.gpu_device_name()))

# Not all sequences are the same length so we are going to PAD them.
pad_x_train =  tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300, padding='post', truncating='pre')
print(pad_x_train.shape)

# We will MASK these, fot that we will create (batch, sequence and features... so we need to add an extra array layer.
# We also need to convert it to TENSOR before masking it.
pad_x_train = np.expand_dims(pad_x_train, -1)
tf_x_train = tf.convert_to_tensor(pad_x_train, dtype='float32')
masking_layer = tf.keras.layers.Masking(mask_value=0.0)
mask_x_train = masking_layer(tf_x_train)
print(mask_x_train._keras_mask)

