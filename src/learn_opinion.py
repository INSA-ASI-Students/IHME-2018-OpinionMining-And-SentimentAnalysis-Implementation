# Tweet, target, Opinion towards

import sklearn.preprocessing

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import hashing_trick

import numpy as np

import string_format

def hash_words(dataset, hash_size=2**(32-1)):
    hashed_dataset = []
    for sentence in dataset:
        hashed_dataset.append(hashing_trick(sentence, hash_size, hash_function='md5'))
    return hashed_dataset

def create_model(max_length):
    keras_model = Sequential()
    keras_model.add(Dense(512, input_shape=(max_length,)))
    keras_model.add(Activation('relu'))
    keras_model.add(Dense(3))
    keras_model.add(Activation('softmax'))

    return keras_model

def main():
    dataset_train = string_format.format_dataset(
        string_format.load_dataset('./src/StanceDataset/train_ingrid.csv', ',')
    )
    dataset_test = string_format.format_dataset(
        string_format.load_dataset('./src/StanceDataset/test_ingrid.csv', ',')
    )

    # Get Train tweets and labels
    num_tweets = 0
    train_tweets = []
    train_labels = []
    for row in dataset_train:
        num_tweets += 1
        train_tweets.append(row['Tweet'])
        train_labels.append(int(row['Opinion Towards'][0]))

    # Convert Train words to numbers
    train_tweets = hash_words(train_tweets)

    # Setting max length to 140 because a tweet can contains only 140 characters
    # so we assume that we cannot have more than 140 words.
    max_length = 140

    # Pad sequence
    np_train_tweets = np.zeros((num_tweets, max_length), dtype=np.int32)
    for i in range(num_tweets):
        np_train_tweets[i, :len(train_tweets[i])] = np.array(train_tweets[i])

    # Get Test tweets and labels
    num_tweets = 0
    test_tweets = []
    test_labels = []
    for row in dataset_test:
        num_tweets += 1
        test_tweets.append(row['Tweet'])
        test_labels.append(int(row['Opinion Towards'][0]))

    # Convert Test words to numbers
    test_tweets = hash_words(test_tweets)

    # Pad sequence
    np_test_tweets = np.zeros((num_tweets, max_length), dtype=np.int32)
    for i in range(num_tweets):
        np_test_tweets[i, :len(test_tweets[i])] = np.array(test_tweets[i])

    # One hot labels
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(3))
    train_labels = label_binarizer.transform(train_labels)
    test_labels = label_binarizer.transform(test_labels)

    keras_model = create_model(max_length)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(train_labels)
    history = keras_model.fit(np_train_tweets, train_labels, batch_size=32, epochs=2)

    score = keras_model.evaluate(np_test_tweets, test_labels, batch_size=32)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return 0

if __name__ == '__main__':
    exit(main())
