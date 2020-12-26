##### Tokenising Text Data ######
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import json

with open ('ThreeMenInABoat.txt', 'r', encoding='utf-8') as file:
    text_string = file.read().replace('\n', ' ')

text_string = text_string.replace('-', '')
print(text_string[:2001])

sentence_strings = text_string.split('.')
print(sentence_strings[20:30])

additional_filters = '—’‘“”'
tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + additional_filters,
                      lower=True,
                      split=' ',
                      char_level=False,
                      oov_token='<UNK>',
                      document_count=0)

# Fit to a "list of strings" or a "list of lists of strings" #
tokenizer.fit_on_texts(sentence_strings)

# View configuration #
tokenizer_config = tokenizer.get_config()
tokenizer_config.keys()
tokenizer_config['word_counts']
tokenizer_config['word_index']

word_counts = json.loads(tokenizer_config['word_counts'])
index_word = json.loads(tokenizer_config['index_word'])
word_index = json.loads(tokenizer_config['word_index'])

# Map sentences to tokens #
print(sentence_strings[:5])
sentence_seq = tokenizer.texts_to_sequences(sentence_strings)
print(type(sentence_seq))
print(sentence_seq[:5])

# Just verifying the mappings are the same as with the JSON dictionaries we imported earlier #
print(word_index['chapter'], word_index['i'])
print(word_index['three'], word_index['invalids'])
print(word_index['sufferings'], word_index['of'], word_index['george'], word_index['and'], word_index['harris'])
print(word_index['a'], word_index['victim'], word_index['to'], word_index['one'], word_index['hundred'], word_index['and'], word_index['seven'], word_index['fatal'], word_index['maladies'])
print(word_index['useful'], word_index['prescriptions'])

# Now we'll do the inverse, mapping indexes to the words #
sentence_seq[0:5]
tokenizer.sequences_to_texts(sentence_seq)[:5]
print(index_word['362'], index_word['8'])
print(index_word['126'], index_word['3362'])
print(index_word['2319'], index_word['6'], index_word['36'], index_word['3'], index_word['35'])
print(index_word['5'], index_word['1779'], index_word['4'], index_word['43'], index_word['363'], index_word['3'], index_word['468'], index_word['3363'], index_word['2320'])
print(index_word['2321'], index_word['3364'])

tokenizer.sequences_to_texts([[92, 104, 241], [152, 169, 53, 2491]])
tokenizer.texts_to_sequences(['i would like goobleydoobly hobbledyho'])
index_word['1']