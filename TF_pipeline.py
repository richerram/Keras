##### Stacked RNNs and Bidirectional wrapper ####
import tensorflow as tf
from tensorflow.keras.datasets import imdb


# A function to load and preprocess the IMDB dataset

def get_and_pad_imdb_dataset(num_words=10000, maxlen=None, index_from=2):

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

(x_train, y_train), (x_test, y_test) = get_and_pad_imdb_dataset(maxlen=250, num_words=5000)
imdb_word_index = get_imdb_word_index(num_words=5000)

max_index_value = max(imdb_word_index.values())
embedding_dim = 16

''' Example using Embedding and LSTM layers.'''
model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=True),
            tf.keras.layers.LSTM(units=32, return_sequences=True),
            tf.keras.layers.LSTM(units=32, return_sequences=False),
            tf.keras.layers.Dense(1, activation='sigmoid')])

'''Example using a Bidirectional layers with different forward and backward layers.'''
model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=True),
            tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(units=8), merge_mode='sum',
                                        backward_layer=tf.keras.layers.GRU(units=8, go_backwards=True)),
            tf.keras.layers.Dense(1, activation='sigmoid')])

##### Example using stacked LSTM and Bidirectional layers. #####
model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=8, return_sequences=True), merge_mode='concat'),
            tf.keras.layers.GRU(units=8, return_sequences=False),
            tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, batch_size=32)

# Predicting #
model.predict(x_test[None, 45, :])
