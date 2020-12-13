import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, imdb

# Easy and straighforward to import
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#~/.keras/datasets/mnist.npz

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#~/.keras/datasets/cifar-10-batches-py

# Another one that requires a bit more arguments
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000, maxlen=100)
''' Where:
num_words = integer or None. Words are ranked by how often they occur (in the training set) 
            and only the num_words most frequent words are kept. Any less frequent word will appear     
            as oov_char value in the sequence data. If None, all words are kept. Defaults to None, so all words are kept.
            
maxlen =    int or None. Maximum sequence length. Any longer sequence will be truncated. Defaults to None, which means no truncation.
'''