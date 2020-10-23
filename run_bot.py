import pandas as pd
import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM

sentence_length = 40

truths = list(pd.read_csv("bristruths_cleaned.csv")["text"])

print("--- Building Vocab ---")

raw_text = ' '.join(truths)
vocab = sorted(set(raw_text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

n_chars = len(raw_text)
n_vocab = len(vocab)

print("--- Build Model ---")
model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, n_vocab)))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("--- Run Model ---")

# load the network weights
filename = "weights-improvement-10-1.3415.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

sentence = "anyone having trouble caring about their"
pattern = [char2idx[char] for char in sentence]

def pat2x(pattern):
	x = np.zeros((1, sentence_length, n_vocab))
	for j, index in enumerate(pattern):
		x[0, j, index] = 1
	return x

x = pat2x(pattern)

print("Seed: ", sentence)

# generate characters
output = ""
for i in range(100):
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = idx2char[index]
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
	x = pat2x(pattern)
	output += result
print(output)
