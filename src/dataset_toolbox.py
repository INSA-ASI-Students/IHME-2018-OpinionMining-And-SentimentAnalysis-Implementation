from nltk.stem.wordnet import WordNetLemmatizer
import nltk
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


def format_tweet(tweet, lemmatizer=WordNetLemmatizer()):
    formatted_tweet = tweet.lower()
    formatted_tweet = remove_semst_hashtag(formatted_tweet)
    formatted_tweet = replace_percent_symbol(formatted_tweet)
    formatted_tweet = replace_money_symbol(formatted_tweet)
    formatted_tweet = remove_useless_symbol(formatted_tweet)
    formatted_tweet = replace_other_symbol(formatted_tweet)
    formatted_tweet = replace_number(formatted_tweet)
    formatted_tweet = replace_mention(formatted_tweet)
    formatted_tweet = trim(formatted_tweet)
    formatted_tweet = verbs_into_infitive(formatted_tweet, lemmatizer)
    return formatted_tweet


def verbs_into_infitive(tweet, lemmatizer=WordNetLemmatizer()):
    formatted_tweet = []
    for word in tweet.split(' '):
        formatted_tweet.append(lemmatizer.lemmatize(word, 'v'))
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


def format_dataset(dataset, lemmatizer=WordNetLemmatizer()):
    formatted_dataset = []
    for row in dataset:
        formatted_row = row
        formatted_row['Tweet'] = format_tweet(row['Tweet'], lemmatizer)
        formatted_dataset.append(formatted_row)
    return formatted_dataset


def init():
    nltk.download('wordnet')
    return WordNetLemmatizer()


def main(file_path, delimiter):
    lemmatizer = init()
    dataset = load_dataset(file_path, delimiter)
    formatted_dataset = format_dataset(dataset, lemmatizer)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 2:
        exit(main(sys.argv[1], sys.argv[2]))
    exit(1)
