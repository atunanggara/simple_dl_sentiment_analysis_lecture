import streamlit as st
import pandas as pd
import numpy as np
import time
import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from numpy import array
from keras.datasets import imdb
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from keras import layers

def main():
    st.title('Deployed Deep Learning Sentiment Analysis')
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    current_dir = os.getcwd()
    current_file = os.path.join(current_dir, "model_simple.h5")
    # test_data[0]
    (_, _), (x_test, y_test) = prep_model(train_data, train_labels,
                                                      test_data, test_labels)
    model_load = load_model(current_file)
    layer_outputs = [layer.output for layer in model_load.layers]
    activation_model = models.Model(inputs = model_load.input,
                                    outputs = layer_outputs)
    result = activation_model.predict(x_test)
    output_test = result[-1]

    st.write('Once we are done tweaking the model, we can use the test data \
              to see how well the model predict them. ')
    st.write('For the test dataset, you can pick \
              a number from 0 to 24,999 below: ')
    decode_test = st.number_input('Insert an index number:',
                                    min_value = 0,
                                    max_value = 24999,
                                    value = 0,
                                    step = 0)
    st.subheader('The particular IMDB review:')
    word_index = imdb.get_word_index()
    st.write(decode_review(decode_test, word_index, test_data))
    if test_labels[decode_test] == 1:
        label_out = "positive"
    else:
        label_out = "negative"
    st.write('The test label for this particular review is: ', label_out)
    st.write('The probability of the review being positive is: ',
              output_test[decode_test][0])


def prep_model(train_data, train_labels, test_data, test_labels):
    x_train = vectorize_sequences(train_data)
    # Our vectorized test data
    x_test = vectorize_sequences(test_data)
    # Our vectorized labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    return (x_train, y_train), (x_test, y_test)


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


def decode_review(index_number, word_index, data):
    # word_index is a dictionary mapping words to an integer index
    # We reverse it, mapping integer indices to words
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # We decode the review; note that our indices were offset by 3
    # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data[index_number]])
    return decoded_review

@st.cache
def load_my_model(MODEL_PATH):
   model = load_model(MODEL_PATH)
   return model

if __name__ == '__main__':
    main()
