##### Stateful RNNs (to retain long sequencese) #####
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Input

# Here is the same RNN, the second one retains the state this is done with the "stateful=True" statement.
gru = Sequential([GRU(5, input_shape=(None,1), name='rnn')])
stateful_gru = Sequential([GRU(5, stateful=True, batch_input_shape=(2, None, 1), name='stateful_rnn')])

'''When using stateful RNNs, it is necessary to supply this argument to the first layer of a Sequential model. 
This is because the model will always assume that each element of every subsequent batch it receives will be 
a continuation of the sequence from the corresponding element in the previous batch.

Another detail is that when defining a model with a stateful RNN using the functional API, you will need to specify 
the batch_shape argument as follows:'''

inputs = Input(batch_shape=(2, None, 1))
outputs = GRU(5, stateful=True, name='stateful_rnn')(inputs)

stateful_gru = Model(inputs, outputs)

# We can Inspect the layers and retrieve the "states" property.
print(gru.get_layer('rnn').states)
print(stateful_gru.get_layer('stateful_rnn').states)

# Now let's use this on a simple data sequence.
sequence_data = tf.constant([
    [[-4.], [-3.], [-2.], [-1.], [0.], [1.], [2.], [3.], [4.]],
    [[-40.], [-30.], [-20.], [-10.], [0.], [10.], [20.], [30.], [40.]]
], dtype=tf.float32)

print (sequence_data.shape)

# Let's pass this sequence through both models.
_1 = gru(sequence_data)
_2 = stateful_gru(sequence_data)

print(gru.get_layer('rnn').states)
# [None]

print(stateful_gru.get_layer('stateful_rnn').states)
'''[<tf.Variable 'stateful_rnn_1/Variable:0' shape=(2, 5) dtype=float32, 
    numpy=array([[ 0.5654753 ,  0.93823063, -0.8363503 ,  0.44468057, -0.63199437],
                [-0.38263583,  1.        , -1.        ,  0.9123156 ,  0.30444482]],
                dtype=float32)>]'''

# You can reset the states using "reset_states()".
stateful_gru.get_layer('stateful_rnn').reset_states()
print(stateful_gru.get_layer('stateful_rnn').states)
'''[<tf.Variable 'stateful_rnn_1/Variable:0' shape=(2, 5) dtype=float32, numpy=
    array([[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]], dtype=float32)>]'''

# Note that passing many subsequences through a stateful RNN gives the same result as passing tue whole sequence at once.
# First we porcess the entire sequence at once.
stateful_gru.get_layer('stateful_rnn').reset_states()
_ = stateful_gru(sequence_data)
print (stateful_gru.get_layer('stateful_rnn').states)
'''[<tf.Variable 'stateful_rnn_1/Variable:0' shape=(2, 5) dtype=float32, numpy=array(
            [[ 0.13407314, -0.24488138,  0.33283684, -0.33388492,  0.11438916],
            [ 0.0450515 ,  0.44803664,  0.8882864 , -0.9989012 , -0.48145008]],dtype=float32)>]'''

# Now we split the sequence into 3 batches and pass 1 by 1.
batch1 = sequence_data[:, :3, :]
batch2 = sequence_data[:, 3:6, :]
batch3 = sequence_data[:, 6:, :]

print(f'First Batch: {batch1}')
print(f'Second Batch: {batch2}')
print(f'Third Batch: {batch3}')

stateful_gru.get_layer('stateful_rnn').reset_states()
_ = stateful_gru(batch1)
_ = stateful_gru(batch2)
_ = stateful_gru(batch3)
stateful_gru.get_layer('stateful_rnn').states
'''[<tf.Variable 'stateful_rnn_1/Variable:0' shape=(2, 5) dtype=float32, numpy= array(
        [[ 0.13407314, -0.24488138,  0.33283684, -0.33388492,  0.11438916],
        [ 0.0450515 ,  0.44803664,  0.8882864 , -0.9989012 , -0.48145008]], dtype=float32)>]'''
