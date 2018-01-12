import csv
from .tweets_formatter import format as format_tweets
import random
from random import shuffle
from collections import OrderedDict
import numpy as np

DELIMITER = ','


def load(filename, delimiter):
    dataset = []
    with open(filename, 'r', encoding='ISO 8859-2') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in spamreader:
            dataset.append(row)
    dataset = _convert(spamreader.fieldnames, dataset)
    return dataset


def _convert(fieldnames, dataset):
    result = {}
    for name in fieldnames:
        result[name] = []
    for row in dataset:
        for name in fieldnames:
            result[name].append(row[name])
    return result


def format(dataset):
    dataset['Tweet'] = format_tweets(dataset['Tweet'])
    return dataset


def save(filename, dataset, delimiter):
    with open(filename, 'w') as csvfile:
        fieldnames = []
        for col in dataset:
            fieldnames.append(col)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(dataset[fieldnames[0]])):
            row = {}
            for col in fieldnames:
                row[col] = dataset[col][i]
            writer.writerow(row)


def data_to_dataset(data):
    return {'Target': list(data[0]), 'Tweet': list(data[1]), 'Opinion Towards': list(data[2]), 'Sentiment': list(data[3]), 'Stance': list(data[4])}


def fusion(train_filename, test_filename, predict_filename):
    dataset_train = format(load(train_filename, DELIMITER))
    dataset_test = format(load(test_filename, DELIMITER))

    target = np.hstack([np.array(dataset_train['Target']), np.array(dataset_test['Target'])])
    tweet = np.hstack([np.array(dataset_train['Tweet']), np.array(dataset_test['Tweet'])])
    opinion = np.hstack([np.array(dataset_train['Opinion Towards']),
                         np.array(dataset_test['Opinion Towards'])])
    sentiment = np.hstack([np.array(dataset_train['Sentiment']),
                           np.array(dataset_test['Sentiment'])])
    stance = np.hstack([np.array(dataset_train['Stance']), np.array(dataset_test['Stance'])])

    data = np.transpose(np.array([target, tweet, opinion, sentiment, stance]))
    np.random.shuffle(data)

    data_length = len(data)
    data_train = data[:round(data_length / 2)]
    data_test = data[round(data_length / 2):]

    if predict_filename == '':
        data_test = data[round(data_length / 2):round(data_length / 2) +
                         round(round(data_length / 2) / 2)]
        data_valid = data[round(data_length / 2) + round(round(data_length / 2) / 2):]
        dataset_valid = data_to_dataset(np.transpose(data_valid))
    else:
        dataset_valid = format(load(predict_filename, DELIMITER))

    dataset_train = data_to_dataset(np.transpose(data_train))
    dataset_test = data_to_dataset(np.transpose(data_test))
    return dataset_train, dataset_test, dataset_valid
