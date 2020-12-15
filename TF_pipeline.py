import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100

# First we will load the Cifar100 set with the "fine" labeling which means more categories to chose from.
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
print(x_train.shape)
print(y_train.shape)

print(y_train[500])
plt.imshow(x_train[500])

import json
with open('cifar100_fine_labels.json', 'r') as fine_labels:
    cifar100_fine = json.load(fine_labels)

print(cifar100_fine[:10])
print(cifar100_fine[41])

examples = x_train[(y_train.T == 30)[0]][:3]
fix, ax = plt.subplots(1,3)
ax[0].imshow(examples[0])
ax[1].imshow(examples[1])
ax[2].imshow(examples[2])

# Now we will do the loading of the same Data Set but with less fine labeling, called COARSE.
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
with open ('cifar100_coarse_labels.json') as coarse_labels:
    cifar100_coarse = json.load(coarse_labels)

examples = x_train[(y_train.T == 10)[0]][:3]
fix, ax = plt.subplots(1,3)
ax[0].imshow(examples[0])
ax[1].imshow(examples[1])
ax[2].imshow(examples[2])


# Next we are doing the import of the IMBD (movies) DataSet.
# This is a bit different since it is a type of sequential data.

from tensorflow.keras.datasets import imdb

(xT_imdb, yT_imdb), (xt_imdb, yt_imdb) = imdb.load_data()
print (xT_imdb[0])
print (yT_imdb[0])

imdb_words = imdb.get_word_index()

text = ""
for i in xT_imdb[0]:
    for k,v in imdb_words.items():
        if i == v:
            text = text + str(k) + " "
print (text)

# Now we are going to load the same data but adding some Keyword arguments:
# "Skip Top" means we are skipping the most recurrent words as we consider them noise.
# "oov_char" means Out-Of-Vocabulary Character and it is the character that will repalce the skipped words, we pick "43" which is "out"
(xT_imdb, yT_imdb), (xt_imdb, yt_imdb) = imdb.load_data(skip_top=50, oov_char=43)
text = ""
for i in xT_imdb[0]:
    for k,v in imdb_words.items():
        if i == v:
            text = text + str(k ) + " "
print (text)
