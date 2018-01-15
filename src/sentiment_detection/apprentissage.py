############### Librairies ###############
import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from utils import metrics
from sklearn.externals import joblib

FILENAME = 'dist/sentiment.model.pkl'


def export_model(model):
    joblib.dump(model, FILENAME)


def import_model():
    return joblib.load(FILENAME)


def get_tweet_sentiment(train_tweet, train_sentiment, test_tweet, test_sentiment):
    X_train_posTweet = []
    X_train_negTweet = []
    X_train_otherTweet = []
    X_test_posTweet = []
    X_test_negTweet = []
    X_test_otherTweet = []

    dim1 = len(train_tweet)
    dim2 = len(test_tweet)

    for i in range(dim1):
        if train_sentiment[i] == 'pos':
            X_train_posTweet.append(train_tweet[i])
        elif train_sentiment[i] == 'neg':
            X_train_negTweet.append(train_tweet[i])
        elif train_sentiment[i] == 'other':
            X_train_otherTweet.append(train_tweet[i])

    for i in range(dim2):
        if test_sentiment[i] == 'pos':
            X_test_posTweet.append(test_tweet[i])
        elif test_sentiment[i] == 'neg':
            X_test_negTweet.append(test_tweet[i])
        elif test_sentiment[i] == 'other':
            X_test_otherTweet.append(test_tweet[i])

    return X_train_posTweet, X_train_negTweet, X_train_otherTweet, X_test_posTweet, X_test_negTweet, X_test_otherTweet


def get_words_sentiment(X_train_posTweet, X_train_negTweet, X_train_otherTweet, X_test_posTweet, X_test_negTweet, X_test_otherTweet):
    X_train_wordPos = []
    X_train_wordNeg = []
    X_train_wordOther = []

    X_test_wordPos = []
    X_test_wordNeg = []
    X_test_wordOther = []

    for (words) in X_train_posTweet:
        words_filtered = [e.lower() for e in words.split() if len(
            e) >= 6]  # 6 : longueur des mots que l'on prends
        X_train_wordPos.append(words_filtered)

    for (words) in X_train_negTweet:
        words_filtered = [e.lower() for e in words.split() if len(
            e) >= 6]  # 6 : longueur des mots que l'on prends
        X_train_wordNeg.append(words_filtered)

    for (words) in X_train_otherTweet:
        words_filtered = [e.lower() for e in words.split() if len(
            e) >= 6]  # 6 : longueur des mots que l'on prends
        X_train_wordOther.append(words_filtered)

    for (words) in X_test_posTweet:
        words_filtered = [e.lower() for e in words.split() if len(
            e) >= 6]  # 6 : longueur des mots que l'on prends
        X_test_wordPos.append(words_filtered)

    for (words) in X_test_negTweet:
        words_filtered = [e.lower() for e in words.split() if len(
            e) >= 6]  # 6 : longueur des mots que l'on prends
        X_test_wordNeg.append(words_filtered)

    for (words) in X_test_otherTweet:
        words_filtered = [e.lower() for e in words.split() if len(
            e) >= 6]  # 6 : longueur des mots que l'on prends
        X_test_wordOther.append(words_filtered)

    return X_train_wordPos, X_train_wordNeg, X_train_wordOther, X_test_wordPos, X_test_wordNeg, X_test_wordOther


def bag_of_words_extraction(words):
    return dict([(word, True) for word in words])


def naive_bayes_classifier(train_set, test_set):
    print('Naive Bayes Classifier')
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    accuracy = (accuracy * 100)
    return accuracy


def bernoulli_nb(train_set, test_set):
    print('Naive Bayes classifier for multivariate Bernoulli models')
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(train_set)
    accuracy = nltk.classify.accuracy(BernoulliNB_classifier, test_set) * 100
    # Naive Bayes classifier for multivariate Bernoulli models. Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features.
    return accuracy


def multinomial_nb(train_set, test_set):
    print('Multinomial Naive Bayes classifier')
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(train_set)
    accuracy = nltk.classify.accuracy(MNB_classifier, test_set) * 100
    # The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
    return accuracy


def logistic_regression(train_set, test_set):
    print('Logistic Regression classifier')
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(train_set)
    accuracy = nltk.classify.accuracy(LogisticRegression_classifier, test_set) * 100
    return accuracy


def sgd_classifier(train_set, test_set):
    print('SGDClassifier classifier')
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(train_set)
    accuracy = nltk.classify.accuracy(SGDClassifier_classifier, test_set) * 100
    return accuracy


def svc(train_set, test_set):
    print('SVC classifier')
    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(train_set)
    accuracy = nltk.classify.accuracy(SVC_classifier, test_set) * 100
    return accuracy


def linear_svc(train_set, test_set):
    print('Linear SVC classifier')
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(train_set)
    accuracy = nltk.classify.accuracy(LinearSVC_classifier, test_set) * 100
    return accuracy


def train(dataset_train, dataset_test):

    test_tweet = dataset_test['Tweet']
    train_tweet = dataset_train['Tweet']

    train_sentiment = dataset_train['Sentiment']
    test_sentiment = dataset_test['Sentiment']

    (X_train_posTweet, X_train_negTweet, X_train_otherTweet, X_test_posTweet, X_test_negTweet,
     X_test_otherTweet) = get_tweet_sentiment(train_tweet, train_sentiment, test_tweet, test_sentiment)
    (X_train_wordPos, X_train_wordNeg, X_train_wordOther, X_test_wordPos, X_test_wordNeg, X_test_wordOther) = get_words_sentiment(
        X_train_posTweet, X_train_negTweet, X_train_otherTweet, X_test_posTweet, X_test_negTweet, X_test_otherTweet)

    positive_features_train = [(bag_of_words_extraction(pos), 'pos') for pos in X_train_wordPos]
    negative_features_train = [(bag_of_words_extraction(neg), 'neg') for neg in X_train_wordNeg]
    neutral_features_train = [(bag_of_words_extraction(other), 'other')
                              for other in X_train_wordOther]

    positive_features_test = [(bag_of_words_extraction(pos), 'pos') for pos in X_test_wordPos]
    negative_features_test = [(bag_of_words_extraction(neg), 'neg') for neg in X_test_wordNeg]
    neutral_features_test = [(bag_of_words_extraction(other), 'other')
                             for other in X_test_wordOther]

    train_set = negative_features_train + positive_features_train + neutral_features_train
    test_set = negative_features_test + positive_features_test + neutral_features_test

    accuracy = multinomial_nb(train_set, test_set)
    print('Good rate in percent: %s  ' % (accuracy))

    pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
    pipe.fit(train_tweet, train_sentiment)

    return pipe

    #success_rate = metrics.success_rate(test_sentiment, y_pred)
    #print('multinomial_nb results good prediction: %s  ' % (success_rate))


def predict(model, dataset):
    y_pred = model.predict(dataset)
    return y_pred.tolist()


def main():
    #dataset = dataset_manager.load('./dataset/train.csv', ',')
    #prediction = predict(dataset['Tweet'])
    #taux_erreur = metrics.success_rate(dataset['Sentiment'], prediction)
    # print(taux_erreur)
    return 0


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '')
    from utils import dataset_manager, metrics
    exit(main())
