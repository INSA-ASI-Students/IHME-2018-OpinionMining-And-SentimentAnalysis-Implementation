import sys

from utils import dependancies
from utils import metrics
from utils import dataset_manager as dm

from sentiment_detection import sentiwordnet as sw
from sentiment_detection import wordnetaffect as wa
from sentiment_detection import apprentissage as ap
from stance_detection import stance_detection as sd
from learn_opinion import neural_network as nn

DELIMITER = ','


def main():
    (action, fusion, train_filename, test_filename, predict_filename, sentiment,
     opinion, stance, output) = define_parameters(sys.argv)

    if action == 'learn':
        return learn(train_filename, test_filename, fusion, sentiment, opinion, stance)
    elif action == 'predict':
        return predict(predict_filename, sentiment, opinion, stance, output)
    return 0


def learn_sentiment(sentiment, dataset_train, dataset_test):
    if sentiment == 'learning':
        return ap.train(dataset_train, dataset_test)
    return None


def learn_opinion(opinion, dataset_train, dataset_test):
    if opinion == 'neural_network':
        return nn.train(dataset_train, dataset_test)
    return None


def learn_stance(stance, dataset_train, dataset_test):
    if stance == 'stance':
        return sd.train(dataset_train, dataset_test)
    return None


def learn(train_filename, test_filename, fusion, sentiment, opinion, stance):
    dataset_train = dm.format(dm.load(train_filename))
    dataset_test = dm.format(dm.load(test_filename))
    if fusion == True:
        dataset_train, dataset_test = dm.fusion(dataset_train, dataset_test)

    prediction = {}

    model_sentiment = learn_sentiment(sentiment, dataset_train, dataset_test)
    if model_sentiment is None:
        print('Invalid sentiment method')
        return 1
    else:
        dm.save_model(model_sentiment, 'sentiment', sentiment)
        prediction['Sentiment'] = predict_sentiment(model_sentiment, dataset_test)
        print_results('Sentiment', sentiment, dataset_valid, prediction['Sentiment'])

    model_opinion = learn_opinion(opinion, dataset_train, dataset_test)
    if model_opinion is None:
        print('Invalid opinion method')
        return 1
    else:
        dm.save_model(model_opinion, 'opinion', opinion)
        prediction['Opinion Towards'] = predict_opinion(model_opinion, dataset_test)
        print_results('Opinion Towards', opinion, dataset_valid, prediction['Opinion Towards'])

    model_stance = learn_stance(stance, dataset_train, dataset_test)
    if stance_prediction is None:
        print('Invalid stance method')
        return 1
    else:
        dm.save_model(model_stance, 'stance', stance)
        prediction['Stance'] = predict_stance(model_stance, dataset_test)
        print_results('Stance', stance, dataset_valid, stance_prediction)

    return 0


def predict_sentiment(sentiment, dataset, model):
    if model == None:
        model = dm.load_model('sentiment', sentiment)
    if sentiment == 'wordnetaffect':
        return wa.predict(dataset['Tweet'])
    elif sentiment == 'sentiwordnet':
        return sw.predict(dataset['Tweet'])
    elif sentiment == 'learning':
        return ap.predict(model, dataset)
    return None


def predict_opinion(opinion, dataset, model):
    if model == None:
        model = dm.load_model('opinion', opinion)
    if opinion == 'neural_network':
        return nn.predict(model, dataset)
    return None


def predict_stance(stance, dataset, model):
    if model == None:
        model = dm.load_model('stance', stance)
    if stance == 'stance':
        return sd.predict_stance(dataset, model)
    return None


def predict(predict_filename, output):
    dataset = dm.format(dm.load(predict_filename))

    sentiment_prediction = predict_sentiment(sentiment, dataset, None)
    if sentiment_prediction is None:
        print('Invalid sentiment method')
        return 1
    else:
        dataset['Sentiment'] = sentiment_prediction

    opinion_prediction = predict_opinion(opinion, dataset, None)
    if opinion_prediction is None:
        print('Invalid opinion method')
        return 1
    else:
        dataset['Opinion Towards'] = opinion_prediction

    stance_prediction = predict_stance(sentiment, dataset, None)
    if stance_prediction is None:
        print('Invalid stance method')
        return 1

    dm.save(output, dataset, DELIMITER)
    return 0


def print_results(column, method, dataset, prediction):
    try:
        truth = dataset[column]
        success_rate = metrics.success_rate(truth, prediction)
        print('%s results: %s  percent' % (method, success_rate * 100))
    except:
        pass


def define_parameters(args):
    action = 'learn'
    fusion = False
    train_filename = './dataset/train.csv'
    test_filename = './dataset/test.csv'
    predict_filename = ''
    sentiment = 'learning'
    opinion = 'neural_network'
    stance = 'stance'
    output = './output.csv'

    for arg in args:
        if arg.startswith('--action='):
            action = arg.split('--action=')[1]
        elif arg.startswith('--sentiment='):
            sentiment = arg.split('--sentiment=')[1]
        elif arg.startswith('--fusion-dataset'):
            fusion = True
        elif arg.startswith('--opinion='):
            opinion = arg.split('--opinion=')[1]
        elif arg.startswith('--stance='):
            stance = arg.split('--stance=')[1]
        elif arg.startswith('--output='):
            output = arg.split('--output=')[1]
    return action, fusion, train_filename, test_filename, predict_filename, sentiment, opinion, stance, output


if __name__ == '__main__':
    exit(main())