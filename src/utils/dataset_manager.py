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
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for i in range(0, len(dataset[fieldnames[0]])):
            row = {}
            for col in fieldnames:
                row[col] = dataset[col][i]
            writer.writerow(row)


def _data_to_dataset(data):
    return {
        'Target': list(data[0]),
        'Tweet': list(data[1]),
        'Opinion Towards': list(data[2]),
        'Sentiment': list(data[3]),
        'Stance': list(data[4])
    }


def fusion(dataset1, dataset2):
    target = np.hstack([np.array(dataset1['Target']),
                        np.array(dataset2['Target'])])
    tweet = np.hstack([np.array(dataset1['Tweet']),
                       np.array(dataset2['Tweet'])])
    opinion = np.hstack([np.array(dataset1['Opinion Towards']),
                         np.array(dataset2['Opinion Towards'])])
    sentiment = np.hstack([np.array(dataset1['Sentiment']),
                           np.array(dataset2['Sentiment'])])
    stance = np.hstack([np.array(dataset1['Stance']),
                        np.array(dataset2['Stance'])])

    data = np.transpose(np.array([target, tweet, opinion, sentiment, stance]))
    np.random.shuffle(data)

    data_length = len(data)
    new_dataset1 = _data_to_dataset(np.transpose(data[:round(data_length / 2)]))
    new_dataset2 = _data_to_dataset(np.transpose(data[round(data_length / 2):]))
    return new_dataset1, new_dataset2
