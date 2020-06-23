import streamlit as st
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from numpy import array
from keras.datasets import imdb
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from keras import layers


def main():
    np.random.seed(23)
    st.title('Simple Deep Learning Model for Sentiment Analysis')
    st.write('I am using an example from Chapter 3 of \
              [Deep learning with Python]\
              (https://www.manning.com/books/deep-learning-with-python) \
               as a proof of concept on how to deploy deep learning \
               on heroku.')
    st.write('In particular, I borrowed code from \
        [this github repository]\
(https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb) \
              and convert it into a streamlit app.')
    st.header('Data source')
    st.write('The data comes from the [IMDB review dataset]\
             (https://ai.stanford.edu/~amaas/data/sentiment/).')
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(\
                                                            num_words=10000)
    word_index = imdb.get_word_index()
    st.write('If you are interested in reading the review, you can pick \
              a number from 0 to 24,999 below: ')
    decode_number = st.number_input('Insert an index number:',
                                    min_value = 0,
                                    max_value = 24999,
                                    value = 0,
                                    step = 0)
    data_choice, label_choice = test_or_train_data(train_data, train_labels,
                                                   test_data, test_labels)
    st.subheader('The particular IMDB review:')
    st.write(decode_review(decode_number, word_index, data_choice))
    if label_choice[decode_number] == 1:
        label_out = "positive"
    else:
        label_out = "negative"
    st.write('The label for this particular review is:', label_out)
    st.header('Simple Deep Learning Model')
    st.write('For default, we are going to use a simple dense two-layer \
              model to understand the classification process:')
    url_dense = 'https://camo.githubusercontent.com/ad8a581c483ced840d4a471329d8654e41883d79/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f626f6f6b2e6b657261732e696f2f696d672f6368332f335f6c617965725f6e6574776f726b2e706e67'
    st.image(url_dense)
    st.write('Here we have 16 nodes as our default number. To picture it \
              in our head, here is an example of a deep learning layer model:')
    url_layers = 'https://miro.medium.com/max/2640/1*_fW76QAM6fXC4v0VSVSdzg.png'
    st.image(url_layers, use_column_width = True)
    st.write('For demonstration purpose, we are going to combine the input- and \
              hidden-layer choices. In reality, they can be separated and \
              given their own options.')
    st.write('If we want to tweak things out, the input and hidden layer node \
              can be modified using the option below.')
    option_unit = st.radio('The number of nodes inside the \
                            input and hidden layer:',
                            ('8', '16', '32', '64'),
                            index = 1)
    st.write('Multiple options are available for the choice of activation \
             functions inside the layers: sigmoid, tanh, and relu:')
    url_act_fn = 'https://datasciencechalktalk.files.wordpress.com/2019/10/07bec-17j5z05cgaaoeb1lnlm-ysg.png'
    st.image(url_act_fn)
    st.write('For default, we used **relu** in the two hidden layers and \
              **sigmoid** for the output layer.')
    st.write('We also have the capacity to change the default value \
              down below.')    # Our vectorized training data
    option_hidden = st.selectbox('Activation function for the input and hidden layer:',
                                 ('relu', 'sigmoid', 'tanh'))
    option_out = st.selectbox('Activation function for the output layer:',
                              ('relu', 'sigmoid', 'tanh'),
                              index=1)
    (x_train, y_train), (x_test, y_test) = prep_model(train_data, train_labels,
                                                      test_data, test_labels)
    st.write('We chose `rmsprop` as our optimizer, \
             `binary_crossentropy` as our loss function, and \
             `accuracy` as our metric of choice. We chose the batch size \
              to be 512. To save time, we will \
              keep these default values and move on to the model computation.')
    st.write('One last component to tweak! We can pick the number of epoch \
             (the amount of times the algorithm cycle through the entirety \
               of the training dataset).')
    model = get_two_model(option_unit, option_hidden, option_out)
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    # y validation
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    # Add a choice to run number of epoch
    epoch_num  = st.slider(label = 'Number of epoch',
                           min_value = 1,
                           max_value = 100,
                           value = 20,
                           step = 1)
    create_progress = st.button('Start the simple model computation...')
    len(test_data)
    history = None
    if create_progress:
        if history is None:
            history = model_work(model, partial_x_train,
                                 partial_y_train, x_val,
                                 y_val, 512, epoch_num)
            plot_loss_train(model, 'loss', 'val_loss')
            plot_metric(model, 'accuracy', 'val_accuracy')
        # model.save('three_dense.hdf5')
        st.success('...and now we\'re done!')
    st.write('if you want to save the model, feel free to do so by hitting \
              the button below.')
    save_button = st.button('Saving the model')
    if save_button:
        model.save('model_simple.h5')
        st.success('The model has been saved as `model_simple.h5` file.')
# def pred_vect_sequence(pred, dimension = 10000):
#     results = np.zeros((1, dimension))
#     for sequence in pred:
#         results[0, sequence] = 1.
#     return results


def prep_model(train_data, train_labels, test_data, test_labels):
    x_train = vectorize_sequences(train_data)
    # Our vectorized test data
    x_test = vectorize_sequences(test_data)
    # Our vectorized labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    return (x_train, y_train), (x_test, y_test)


def plot_metric(model, str_metric, str_val_metric):
    metric = model.history.history[str_metric]
    val_metric = model.history.history[str_val_metric]
    epochs = range(1, len(metric) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(epochs),
                             y = metric,
                             mode = 'lines',
                             name = str_metric))
    fig.add_trace(go.Scatter(x = list(epochs),
                             y = val_metric,
                             mode = 'markers',
                             name = str_val_metric))
    fig.update_layout(title = 'Training and validation Metric',
                      xaxis_title = 'Epochs',
                      yaxis_title = str_metric,
                      plot_bgcolor = 'white',
                      showlegend = False)
    return st.plotly_chart(fig, use_container_width = True)


def plot_loss_train(model, str_loss, str_val_loss):
    loss = model.history.history[str_loss]
    val_loss = model.history.history[str_val_loss]
    epochs = range(1, len(loss) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(epochs),
                             y = loss,
                             mode = 'lines',
                             name = str_loss))
    fig.add_trace(go.Scatter(x = list(epochs),
                             y = val_loss,
                             mode = 'markers',
                             name = str_val_loss))
    fig.update_layout(title = 'Training and validation loss',
                      xaxis_title = 'Epochs',
                      yaxis_title = str_loss,
                      plot_bgcolor = 'white',
                      showlegend = False)
    return st.plotly_chart(fig, use_container_width = True)





def model_work(model, x_train, y_train, x_val, y_val, size_batch, num_epoch):
    model.fit(x_train,
              y_train,
              batch_size=size_batch,
              epochs=num_epoch,
              validation_data=(x_val, y_val)
              )
    return model


# def get_one_model():
#     model = models.Sequential()
#     model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model


def get_two_model(node_input,act_inp_choice,act_out_choice):
    node_input = int(node_input)
    model = models.Sequential()
    model.add(layers.Dense(node_input, activation = act_inp_choice,
                           input_shape=(10000,)))
    model.add(layers.Dense(node_input, activation = act_inp_choice))
    model.add(layers.Dense(1, activation = act_out_choice))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# def get_three_model():
#     model = models.Sequential()
#     model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
#     model.add(layers.Dense(16, activation='relu'))
#     model.add(layers.Dense(16, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


def test_or_train_data(train_data, train_labels, test_data, test_labels):
    choice_testtrain = {
        "train_data": "training data",
        "test_data" : "testing data"
    }
    radio_choice = st.radio(
        'Would you be interested in the testing data or training data?',
        ('train_data', 'test_data'),
        format_func = choice_testtrain.get
        )
    if radio_choice == 'train_data':
        data_choice = train_data
        label_choice = train_labels
    else:
        data_choice = test_data
        label_choice = test_labels
    return data_choice, label_choice


def decode_review(index_number, word_index, data):
    # word_index is a dictionary mapping words to an integer index
    # We reverse it, mapping integer indices to words
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # We decode the review; note that our indices were offset by 3
    # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data[index_number]])
    return decoded_review


if __name__ == '__main__':
    main()
