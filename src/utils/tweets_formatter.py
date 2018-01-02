from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
import sys
from . import dataset_manager


def format_tweet(tweet, lemmatizer=WordNetLemmatizer()):
    formatted_tweet = tweet.lower()
    formatted_tweet = remove_semst_hashtag(formatted_tweet)
    formatted_tweet = replace_percent_symbol(formatted_tweet)
    formatted_tweet = replace_money_symbol(formatted_tweet)

    # Ne pas appliquer avec sentiment analysis
    formatted_tweet = replace_contractions(formatted_tweet)
    formatted_tweet = verbs_into_infitive(formatted_tweet, lemmatizer)

    formatted_tweet = remove_useless_symbol(formatted_tweet)
    formatted_tweet = replace_other_symbol(formatted_tweet)
    formatted_tweet = replace_number(formatted_tweet)
    formatted_tweet = replace_mention(formatted_tweet)
    formatted_tweet = trim(formatted_tweet)
    return formatted_tweet


def verbs_into_infitive(tweet, lemmatizer=WordNetLemmatizer()):
    formatted_tweet = []
    for word in tweet.split(' '):
        formatted_tweet.append(lemmatizer.lemmatize(word, 'v'))
    return ' '.join(formatted_tweet)


def remove_semst_hashtag(str):
    return str.replace('#semst', '')


def replace_percent_symbol(str):
    return str.replace('%', ' percent')


def replace_other_symbol(str):
    str = str.replace('&', 'and')
    str = str.replace('@', 'at')
    return str


def replace_money_symbol(str):
    str = str.replace('$', 'dollars ')
    str = str.replace('€', ' euros')
    return str


def replace_contractions(str):
    str = str.replace('ma\'am', 'madam')
    str = str.replace('o\'clock', 'of the clock')
    str = str.replace('\'m', ' am')
    str = str.replace('\'s', ' is')
    str = str.replace('\'ll', ' will')
    str = str.replace('n\'t', ' not')
    str = str.replace('\'ve', ' have')
    str = str.replace('\'d', ' had')
    str = str.replace('\'cause', ' because')
    str = str.replace(' ca  ', ' can ')
    str = str.replace(' sha  ', ' shall ')
    return str


def remove_determinants(str):
    # word type list: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
    tags = nltk.pos_tag(nltk.word_tokenize(str))
    result = []
    for tag in tags:
        if tag[1] != 'DT':
            result.append(tag[0])
    return ' '.join(result)


def remove_useless_symbol(str):
    return re.sub(r'\.|,|;|#|\?|:|-|>|\(|\)|\'|"|!|\*|<|>|_|=|\+|/', ' ', str)


def replace_number(str):
    return re.sub(r'\d+', 'number', str)


def replace_mention(str):
    str = re.sub(r'@[a-z]+', 'someone', str)
    str = re.sub(r'(someone )+', 'someone ', str)
    return str


def trim(str):
    return re.sub(r' +', ' ', str.strip())


def format(dataset, lemmatizer=WordNetLemmatizer()):
    formatted_tweets = []
    for row in dataset:
        formatted_tweets.append(format_tweet(row, lemmatizer))
    return formatted_tweets


def main(file_path, delimiter):
    lemmatizer = WordNetLemmatizer()
    dataset = dataset_manager.load(file_path, delimiter)
    formatted_tweets = format(dataset['Tweet'], lemmatizer)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 2:
        exit(main(sys.argv[1], sys.argv[2]))
    exit(1)
