# Recurrent Neural Nets (RNN) #
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU

''' # Example of a RNN model
model = Sequential([
    Embedding(1000, 32, input_length=64),   # (None, 64, 32)
    SimpleRNN(64, activation='tanh'),       # (None, 64) .... we could also use LSTM or GRU layers here.
    Dense(5, activation='softmax')])        # (None, 5)'''

# SimpleRNN layer and test
simplernn_layer = SimpleRNN(units=16)

# it only returns the final output.
sequence = tf.constant([[[1.,1.], [2.,2.],[56.,-100.]]])
layer_output = simplernn_layer(sequence)
print (layer_output)


# A function to load and preprocess the IMDB dataset

def get_and_pad_imdb_dataset(num_words=10000, maxlen=None, index_from=2):
    from tensorflow.keras.datasets import imdb

    # Load the reviews
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',
                                                          num_words=num_words,
                                                          skip_top=0,
                                                          maxlen=maxlen,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=index_from)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                            maxlen=None,
                                                            padding='pre',
                                                            truncating='pre',
                                                            value=0)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                           maxlen=None,
                                                           padding='pre',
                                                           truncating='pre',
                                                           value=0)
    return (x_train, y_train), (x_test, y_test)

# A function to get the dataset word index

def get_imdb_word_index(num_words=10000, index_from=2):
    imdb_word_index = tf.keras.datasets.imdb.get_word_index(
                                        path='imdb_word_index.json')
    imdb_word_index = {key: value + index_from for
                       key, value in imdb_word_index.items() if value <= num_words-index_from}
    return imdb_word_index

(x_train, y_train), (x_test, y_test) = get_and_pad_imdb_dataset(maxlen=250)
imdb_word_index = get_imdb_word_index()

# To make the Embedding layer we need to get the maximum index value.
max_index_value = max(imdb_word_index.values())
embedding_dim = 16

model = Sequential([
    Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=True),
    LSTM(units=16),
    Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), )

inv_imdb_index = {v:k for k,v in imdb_word_index.items()}
print(*[inv_imdb_index[index] for index in x_test[0] if index>2])

# Let's make a prediction
model.predict(x_test[None, 0, :]) # we see that it gives a 99.2% probability that this is a positive review.