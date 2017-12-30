import csv
from .tweets_formatter import format as format_tweets


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
