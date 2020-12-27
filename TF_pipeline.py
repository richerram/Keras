# Embedding Projector (using IMDB dataset) #
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


# Function to load and pre-porcess the dataset.
def get_and_pad_imdb_dataset(num_words=1000, maxlen=None, index_from=2):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words, skip_top=0,
                                                          maxlen=maxlen, start_char=1,
                                                          oov_char=2, index_from=index_from)
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='pre', truncating='pre', value=0)

    x_test = pad_sequences(x_test, maxlen=maxlen, padding='pre', truncating='pre', value=0)

    return (x_train, y_train), (x_test, y_test)

# Function to the dataset word index.
def get_imdb_word_index(num_words=1000, index_from=2):
    imdb_word_index = imdb.get_word_index()
    imdb_word_index = {k:v + index_from for k,v in imdb_word_index.items() if v + index_from < num_words}
    return imdb_word_index

# Loading
(x_train, y_train), (x_test, y_test) = get_and_pad_imdb_dataset()
imdb_word_index = get_imdb_word_index()

# Swap key:value in dictionary and print to check out.
inv_imdb_word_index = {v:k for k,v in imdb_word_index.items()}
print(*[inv_imdb_word_index[i] for i in x_train[100] if i>2])

##### Let's EMBED #####
# First we need to know the maximun value we can index to.
max_index_value = max(imdb_word_index.values())
print(max_index_value)

embedding_dim = 16

# Building the model.
model = Sequential([
    Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=False),
    GlobalAveragePooling1D(),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_steps=20)

# Plotting the results
plt.style.use('ggplot')

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14,5))
plt.plot(epochs, acc, marker='.', label='Training acc')
plt.plot(epochs, val_acc, marker='.', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Classification Accuracy')
plt.legend(loc='lower right')
plt.ylim(0, 1);


###### Now we export data to be used in the EMBEDDING PROJECTOR #####

    weights = model.layers[0].get_weights()[0] # extracting the weights

    import io
    from os import path

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')

    k = 0

    for word, token in imdb_word_index.items():
        if k != 0:
            out_m.write('\n')
            out_v.write('\n')

        out_v.write('\t'.join([str(x) for x in weights[token]]))
        out_m.write(word)
        k += 1

    out_m.close()
    out_v.close()