# Tweet, target, Opinion towards

import sklearn.preprocessing

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import hashing_trick

import numpy as np

import dataset_toolbox

VOCAB_SIZE = 2**20

def hash_words(dataset, hash_size=VOCAB_SIZE):
    hashed_dataset = []
    for sentence in dataset:
        hashed_dataset.append(hashing_trick(sentence, hash_size, hash_function='md5'))
    return hashed_dataset

def create_model(vocab_size, embed_output_dim):
    keras_model = Sequential()
    keras_model.add(Embedding(vocab_size, embed_output_dim))
    keras_model.add(LSTM(100, recurrent_dropout=0.2))
    keras_model.add(Dense(3, activation='relu'))

    return keras_model

def shape_data(dataset, subjects):
    """
    Transform a 2D numpy array of dimension N, M
    to a 3D numpy of dimension N, W, F

    N : Number of elements
    W : Sequence length
    F : Number of feature in a sequence
    """
    pass

def main():
    dataset_train = dataset_toolbox.format_dataset(
        dataset_toolbox.load_dataset('./src/StanceDataset/train_ingrid.csv', ',')
    )
    dataset_test = dataset_toolbox.format_dataset(
        dataset_toolbox.load_dataset('./src/StanceDataset/test_ingrid.csv', ',')
    )

    # Get Train tweets and labels
    num_tweets = 0
    train_tweets = []
    train_subjects = []
    train_labels = []
    for row in dataset_train:
        num_tweets += 1
        train_tweets.append(row['Tweet'])
        train_subjects.append(row['Target'])
        train_labels.append(int(row['Opinion Towards'][0]))

    # Convert Train words to numbers
    train_tweets = hash_words(train_tweets)
    train_subjects = hash_words(train_subjects)

    # Setting max length to 140 because a tweet can contains only 140 characters
    # so we assume that we cannot have more than 140 words.
    max_length = 140

    # Pad sequence
    np_train_tweets = np.zeros((num_tweets, max_length), dtype=np.int32)
    for i in range(num_tweets):
        np_train_tweets[i, :len(train_tweets[i])] = np.array(train_tweets[i])
    # np_train_tweets = sequence.pad_sequences(train_tweets, maxlen=max_length)

    # Get Test tweets and labels
    num_tweets = 0
    test_tweets = []
    test_subjects = []
    test_labels = []
    for row in dataset_test:
        num_tweets += 1
        test_tweets.append(row['Tweet'])
        test_subjects.append(row['Target'])
        test_labels.append(int(row['Opinion Towards'][0]))

    # Convert Test words to numbers
    test_tweets = hash_words(test_tweets)
    test_subjects = hash_words(test_subjects)

    # Pad sequence
    np_test_tweets = np.zeros((num_tweets, max_length), dtype=np.int32)
    for i in range(num_tweets):
        np_test_tweets[i, :len(test_tweets[i])] = np.array(test_tweets[i])
    # np_test_tweets = sequence.pad_sequences(test_tweets, maxlen=max_length)

    # One hot labels
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(3))
    train_labels = label_binarizer.transform(train_labels)
    test_labels = label_binarizer.transform(test_labels)

    embedding_vector_length = 32
    keras_model = create_model(VOCAB_SIZE, embedding_vector_length)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    history = keras_model.fit(np_train_tweets, train_labels, batch_size=64, epochs=3)

    score = keras_model.evaluate(np_test_tweets, test_labels)
    print('Test score:', score[0])
    print('Test accuracy:', score[1] * 100, '%')
    return 0

if __name__ == '__main__':
    exit(main())

# 1er modèle :
# Apprentissage avec un réseau RNN des tweets
# pour prendre en compte dans l'apprentissage
# les séquences de mots
# Input
# LSTM (100)
# Dense(3)
# Activation('sigmoid')
#
# Résultat : Loss : 0.62 ; Test accuracy : 58.69%

# 2ème modèle :
# Ajout de Dropout pour éviter le surapprentissage
# Input
# LSTM (100)
# Dropout (0.5)
# Dense (3)
# Activation('relu')
#
# Résultat : Loss : 0.62 ; Test accuracy : 58.69%
# Le résultat est le même qu'avec le 1er modèle : Modèle surappris?

# 3ème modèle :
# Dropout au niveau des cellules LSTM
# Input
# LSTM (100, recurrent_dropout=0.2)
# Dense (3)
# Activation('relu')
#
# Résultat : Loss : 0.62 ; Test accuracy : 58.69%
# Le résultat est le même qu'avec le 1er modèle : Modèle surappris?
