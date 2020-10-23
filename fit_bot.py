import pandas as pd
import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

truths = list(pd.read_csv("bristruths_cleaned.csv")["text"])

print("--- Building Vocab ---")

raw_text = ' '.join(truths)
vocab = sorted(set(raw_text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

n_chars = len(raw_text)
n_vocab = len(vocab)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

print("--- Creating Sentences ---")

# model input - 40 char
# model output - next char

sentence_length = 40
sentences = []
next_chars = []
for truth in truths:
    for i in range(0, len(truth) - sentence_length):
        sentences.append(truth[i: i + sentence_length])
        next_chars.append(truth[i + sentence_length])

c = list(zip(sentences, next_chars))
random.shuffle(c)
sentences, next_chars = zip(*c)

print('Total Sentences:', len(sentences))

print("--- Vectorising Sentences ---")

x = np.zeros((len(sentences), sentence_length, n_vocab))
y = np.zeros((len(sentences), n_vocab))

for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, char2idx[char]] = 1
    y[i, char2idx[next_chars[i]]] = 1

print("--- Build Model ---")
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print("--- Fit Model ---")
model.fit(x, y, epochs=10, batch_size=64, callbacks=callbacks_list)
