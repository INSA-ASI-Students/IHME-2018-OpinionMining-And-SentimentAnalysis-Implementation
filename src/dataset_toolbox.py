import numpy as np
import csv
import re
import sys


def load_dataset(filename, delimiter):
    dataset = []
    with open(filename, 'r', encoding='ISO 8859-2') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in spamreader:
            dataset.append(row)
    return dataset


def format_tweet(tweet):
    formatted_tweet = tweet.lower()
    formatted_tweet = remove_semst_hashtag(formatted_tweet)
    formatted_tweet = replace_percent_symbol(formatted_tweet)
    formatted_tweet = replace_money_symbol(formatted_tweet)
    formatted_tweet = remove_useless_symbol(formatted_tweet)
    formatted_tweet = replace_other_symbol(formatted_tweet)
    formatted_tweet = replace_number(formatted_tweet)
    formatted_tweet = replace_mention(formatted_tweet)
    formatted_tweet = trim(formatted_tweet)
    return formatted_tweet


def remove_semst_hashtag(str):
    return str.replace('#semst', '')


def replace_percent_symbol(str):
    return str.replace('%', ' percent')


def replace_other_symbol(str):
    return str.replace('&', 'and')


def replace_money_symbol(str):
    str = str.replace('$', 'dollars ')
    str = str.replace('€', ' euros')
    return str


def remove_useless_symbol(str):
    return re.sub(r'\.|,|;|#|\?|:|-|>|\(|\)|\'|"|!|\*|<|>|_|=|\+', '', str)


def replace_number(str):
    return re.sub(r'\d+', 'number', str)


def replace_mention(str):
    str = re.sub(r'@[a-z]+', 'someone', str)
    str = re.sub(r'(someone )+', 'someone ', str)
    return str


def trim(str):
    return re.sub(r' +', ' ', str.strip())


def format_dataset(dataset):
    formatted_dataset = []
    for row in dataset:
        formatted_row = row
        formatted_row['Tweet'] = format_tweet(row['Tweet'])
        formatted_dataset.append(formatted_row)
    return formatted_dataset


def main(file_path, delimiter):
    dataset = load_dataset(file_path, delimiter)
    formatted_dataset = format_dataset(dataset)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 2:
        exit(main(sys.argv[1], sys.argv[2]))
    exit(1)
