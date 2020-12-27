# Embedding layer (tf.constant) #
import tensorflow as tf
from tensorflow.keras.layers import Embedding

print ('GPU names {}'.format(tf.test.gpu_device_name()))

embedding_layer = Embedding(input_dim=501, output_dim=16)

# remember Embedding layer will expect an input with the shape (batch, sequence, features)
sequence_indices = tf.constant([[[0], [1], [5], [500]]])
sequence_embeddings = embedding_layer(sequence_indices)
print(sequence_embeddings) # see how there are the same number of features but embedded into a 16 dim vectors

# You can retrieve the embedding weights vectors using "get_weights()"
print(embedding_layer.get_weights()[0])

# and you can access the vector for each index too.
print(embedding_layer.get_weights()[0][14])

# We can mask the input too.
masking_embedding_layer = Embedding(input_dim=501, output_dim=16, mask_zero=True)
masked_sequence = masking_embedding_layer(sequence_indices)
print(masked_sequence._keras_mask) # Notice how the "0" has been marked as False.

# NOTE: Embedding layer can only go at the beggining of a model.


