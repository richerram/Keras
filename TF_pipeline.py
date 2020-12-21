##### Processing sequence data (Padding and Masking)#####

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking
import numpy as np

test_input = [[4, 12, 33, 18],
            [63, 23, 54, 30, 19, 3],
            [43, 37, 11, 33, 15]]

preprocessed_data = pad_sequences(test_input, padding='pre')
''' [[ 0  0  4 12 33 18]
    [63 23 54 30 19  3]
    [ 0 43 37 11 33 15]]'''
preprocessed_data = pad_sequences(test_input, padding='post')
''' [[ 4 12 33 18  0  0]
    [63 23 54 30 19  3]
    [43 37 11 33 15  0]]'''
preprocessed_data = pad_sequences(test_input, padding='post', maxlen=5)
''' [[ 4 12 33 18  0]
    [23 54 30 19  3]
    [43 37 11 33 15]]'''
preprocessed_data = pad_sequences(test_input, padding='post', maxlen=5, truncating='post', value=-1)
''' [[ 4 12 33 18 -1]
    [63 23 54 30 19]
    [43 37 11 33 15]]'''

# Now we are going to use the Masking Layer if we want to ignore the sections that are padded (the "0's" added.
preprocessed_data = pad_sequences(test_input, padding='post')

masking_layer = Masking(mask_value=0)
preprocessed_data = preprocessed_data[..., np.newaxis] # the Layer is expecting a 3-dimensional input so... (batch_size, seq_length, features).
''' [[[ 4] [12] [33] [18] [ 0] [ 0]]
    [[63] [23] [54] [30] [19] [ 3]]
    [[43] [37] [11] [33] [15] [ 0]]]'''

masked_input = masking_layer(preprocessed_data)
''' tf.Tensor(  [[[ 4] [12] [33] [18] [ 0] [ 0]]
                [[63] [23] [54] [30] [19] [ 3]]
                [[43] [37] [11] [33] [15] [ 0]]], shape=(3, 6, 1), dtype=int32)'''

masked_input._keras_mask
''' <tf.Tensor: shape=(3, 6), dtype=bool, numpy= array( 
[[ True,  True,  True,  True, False, False],
[ True,  True,  True,  True,  True,  True],
[ True,  True,  True,  True,  True, False]])>'''


# All this easily works on lists of lists
test_input = [[[2,1], [3,3]],
              [[4,3], [2,4], [1,1]]]

preprocessed_data = pad_sequences(test_input, padding='post')
''' [[[2 1] [3 3] [0 0]]
    [[4 3] [2 4] [1 1]]]'''


