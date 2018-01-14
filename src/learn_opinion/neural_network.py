# Tweet, target, Opinion towards

import sklearn.preprocessing

from keras.models import Model
from keras.layers import Input, Concatenate, Dense,\
    LSTM, Dropout, Embedding,\
    Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import hashing_trick

import numpy as np

from utils import dataset_manager as dp

# Setting max length to 140 because a tweet can contains only 140 characters
# so we assume that we cannot have more than 140 words.
MAX_SENTENCE_LENGTH = 140
MAX_SUBJECT_LENGTH = 10
# Setting the max number of existing words
VOCAB_SIZE = 2**20
# Batch size
BATCH_SIZE = 64

# Binarizer (To get one hot encoded labels)
LABEL_BINARIZER = sklearn.preprocessing.LabelBinarizer()
LABEL_BINARIZER.fit(range(3))


def hash_words(dataset, hash_size=VOCAB_SIZE):
    hashed_dataset = []
    for sentence in dataset:
        hashed_dataset.append(hashing_trick(sentence, hash_size, hash_function='md5'))
    return hashed_dataset


def format_tweets(tweets):
    formatted_tweets = []
    for sentence in tweets:
        formatted_tweets.append(' '.join(sentence))
    return tweets


def create_model(sentence_length, subject_length, vocab_size, embed_output_dim):
    tweets = Input(shape=(sentence_length,), name='tweets')
    subjects = Input(shape=(subject_length,), name='subjects')
    concat = Concatenate(axis=1)([tweets, subjects])

    embed = Embedding(vocab_size, embed_output_dim)(concat)
    conv_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embed)
    pool_1 = MaxPooling1D(pool_size=2)(conv_1)
    conv_2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(pool_1)
    pool_2 = MaxPooling1D(pool_size=2)(conv_2)
    rnn = LSTM(100, recurrent_dropout=0.2)(pool_2)
    dropout = Dropout(0.2)(rnn)
    output = Dense(3, activation='sigmoid')(dropout)

    keras_model = Model(inputs=[tweets, subjects], outputs=output)
    return keras_model


def format_dataset(dataset):
    # Extract data
    tweets = format_tweets(dataset['Tweet'])
    subjects = dataset['Target']

    # Convert words to numbers
    tweets = np.array(hash_words(tweets))
    subjects = np.array(hash_words(subjects))

    # Pad dataset to have the same size
    tweets = sequence.pad_sequences(tweets, maxlen=MAX_SENTENCE_LENGTH)
    subjects = sequence.pad_sequences(subjects, maxlen=MAX_SUBJECT_LENGTH)

    return tweets, subjects


def format_labels(dataset):
    labels = []
    for row in dataset['Opinion Towards']:
        labels.append(int(row[0]))

    # One hot labels
    labels = LABEL_BINARIZER.transform(labels)

    return labels


def train(dataset_train, dataset_test):
    # Get Train tweets, subjects and labels
    train_tweets, train_subjects = format_dataset(dataset_train)
    train_labels = format_labels(dataset_train)

    # Get Test tweets, subjects and labels
    test_tweets, test_subjects = format_dataset(dataset_test)
    test_labels = format_labels(dataset_test)

    # Train
    embedding_vector_length = 32
    keras_model = create_model(
        MAX_SENTENCE_LENGTH,
        MAX_SUBJECT_LENGTH,
        VOCAB_SIZE,
        embedding_vector_length
    )
    keras_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    keras_model.fit(
        {'tweets': train_tweets, 'subjects': train_subjects},
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=10
    )

    # Test
    score = keras_model.evaluate(
        {'tweets': test_tweets, 'subjects': test_subjects},
        test_labels
    )
    print('Test score:', score[0])
    print('Test accuracy:', score[1] * 100, '%')

    return keras_model


def predict(model, dataset):
    tweets, subjects = format_dataset(dataset)
    prediction = model.predict(
        {'tweets': tweets, 'subjects': subjects},
        batch_size=BATCH_SIZE
    )
    prediction = np.argmax(prediction, axis=1)

    return prediction


def main():
    dataset_train = dp.format(dp.load('./dataset/train.csv', ','))
    dataset_test = dp.format(dp.load('./dataset/test.csv', ','))

    model = train(dataset_train, dataset_test)
    predict(model, dataset_test)
    return 0


if __name__ == '__main__':
    exit(main())

# 1er modèle :
# Apprentissage avec un réseau RNN des tweets
# pour prendre en compte dans l'apprentissage
# les séquences de mots
#
# loss : categorical_crossentropy ; optimizer : adam
#
# Input
# LSTM (100)
# Dense(3)
# Activation('sigmoid')
#
# Résultat : Loss : 0.62 ; Test accuracy : 58.69%

# 2ème modèle :
# Ajout de Dropout pour éviter le surapprentissage
#
# loss : categorical_crossentropy ; optimizer : adam
#
# Input
# LSTM (100)
# Dropout (0.5)
# Dense (3)
# Activation('relu')
#
# Résultat : Loss : 0.62 ; Test accuracy : 58.69%
# Le résultat est le même qu'avec le 1er modèle : Modèle surappris?

# 3ème modèle :
#
# loss : categorical_crossentropy ; optimizer : adam
#
# Dropout au niveau des cellules LSTM
# Input
# LSTM (100, recurrent_dropout=0.2)
# Dense (3)
# Activation('sigmoid')
#
# Résultat : Loss : 0.62 ; Test accuracy : 58.69%
# Le résultat est le même qu'avec les précédents modèles : Modèle surappris,
# sûrement un manque de données. La loss utilisée peut aussi être l'une des raisons.

# 4ème modèle :
#
# loss : binary_crossentropy ; optimizer : adam
#
# Essai avec couche convolutionnelles
# Input
# Convolution
# MaxPooling
# LSTM (100, recurrent_dropout=0.2)
# Dense (3)
# Activation('sigmoid')
#
# Résultat : Loss : 0.44 ; Test accuracy : 73.65%
# Le résultat est mieux après changement de la loss,
# mais le modèle semble toujours surappris.

# 5ème modèle :
#
# loss : binary_crossentropy ; optimizer : adam
#
# Essai avec SimpleRNN à la place d'une couche LSTM + Dropout
# Input
# Convolution
# MaxPooling
# SimpleRNN (100, recurrent_dropout=0.2)
# Dropout (0.2)
# Dense (3)
# Activation('sigmoid')
#
# Résultat : Loss : 0.01 ; Test accuracy : 72.54%
# Le résultat obtenu est beaucoup plus intéressant,
# le modèle semble ne plus surapprendre et apprend
# mieux les données d'apprentissage.

# 6ème modèle :
#
# loss : binary_crossentropy ; optimizer : adam
#
# Essai avec GRU (Gated Recurrent Unit)
# Input
# Convolution
# MaxPooling
# GRU (100, recurrent_dropout=0.2)
# Dropout (0.2)
# Dense (3)
# Activation('sigmoid')
#
# Résultat : Loss : 0.03 ; Test accuracy : 72.75%
# Pas de surapprentissage non plus

# 7ème modèle :
#
# loss : binary_crossentropy ; optimizer : adam
#
# Apprentissage avec le sujet comme indication supplémentaire
#
# Input
# Convolution (32, 3)
# MaxPooling (2)
# Convolution (64, 3)
# MaxPooling (2)
# LSTM (100, recurrent_dropout=0.2)
# Dropout (0.2)
# Dense (3)
# Activation('sigmoid')
#
# Résultat : Loss : 0.03 ; Test accuracy : 72.95%
# Pas de surapprentissage. Le fait que le résultat en test ne soit pas
# vraiment élevé est dû au fait qu'il y a peu de données sur lesquelles
# entrainer le réseau de neurones
