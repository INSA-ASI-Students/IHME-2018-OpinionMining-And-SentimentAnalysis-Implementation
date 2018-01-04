# Tweet, target, Opinion towards

import sklearn.preprocessing

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, Embedding, Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import hashing_trick

import numpy as np

from utils import dataset_manager as dp

# Setting max length to 140 because a tweet can contains only 140 characters
# so we assume that we cannot have more than 140 words.
MAX_SENTENCE_LENGTH = 140
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
        hashed_dataset.append(hashing_trick(' '.join(sentence), hash_size, hash_function='md5'))
    return hashed_dataset


def create_model(vocab_size, embed_output_dim):
    keras_model = Sequential()
    keras_model.add(Embedding(vocab_size, embed_output_dim))
    keras_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    keras_model.add(MaxPooling1D(pool_size=2))
    keras_model.add(SimpleRNN(100, recurrent_dropout=0.2))
    keras_model.add(Dropout(0.2))
    keras_model.add(Dense(3, activation='sigmoid'))

    return keras_model

def format_dataset(dataset):
    # Extract data
    tweets = dataset['Tweet']
    subjects = dataset['Target']

    # Convert words to numbers
    tweets = hash_words(tweets)
    subjects = hash_words(subjects)

    # Pad dataset to have the same size
    np_tweets = sequence.pad_sequences(tweets, maxlen=MAX_SENTENCE_LENGTH)

    return tweets, subjects, np_tweets

def format_labels(dataset):
    labels = []
    for row in dataset['Opinion Towards']:
        labels.append(int(row[0]))
    print(labels)

    # One hot labels
    labels = LABEL_BINARIZER.transform(labels)

    return labels

def train(dataset_train, dataset_test):
    # Get Train tweets, subjects and labels
    train_tweets, train_subjects, np_train_tweets = format_dataset(dataset_train)
    train_labels = format_labels(dataset_train)

    # Get Test tweets, subjects and labels
    test_tweets, test_subjects, np_test_tweets = format_dataset(dataset_test)
    test_labels = format_labels(dataset_test)

    # Train
    embedding_vector_length = 32
    keras_model = create_model(VOCAB_SIZE, embedding_vector_length)
    keras_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = keras_model.fit(np_train_tweets, train_labels, batch_size=BATCH_SIZE, epochs=10)

    # Test
    score = keras_model.evaluate(np_test_tweets, test_labels)
    print('Test score:', score[0])
    print('Test accuracy:', score[1] * 100, '%')

    return keras_model

def predict(model, dataset):
    tweets, subjects, np_tweets = format_dataset(dataset)
    prediction = model.predict(np_tweets, batch_size=BATCH_SIZE)
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