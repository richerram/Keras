# Embedding Layer example #
import tensorflow as tf
from tensorflow.keras.layers import Embedding
import numpy as np


# The Embedding layer takes two inputs, the vocabulary size and the dimensional space, so in the following example,
# each word in a vocabulary of 1,000 words will be embedded into a space of 32 dimensions.
# We can also add a Masking layer which will do padding into our sequence

emb_layer = Embedding(1000, 32, input_length=64, mask_zero=True)

# Toy example with random input.
test_input = np.random.randint(1000, size=(16,64)) # Batch of 16 with each being of length 64.

emb_inputs = emb_layer(test_input) # This will create an embedded into 32 dimensions Tensor of shape (16, 64, 32)
print(emb_inputs._keras_mask)


