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
    (filename, sentiment, opinion, stance, output) = define_parameters(sys.argv)
    dataset = dm.format(dm.load(filename, DELIMITER))

    sentiment_prediction = predict_sentiment(sentiment, dataset)
    if sentiment_prediction is None:
        print('Invalid sentiment method')
        return 1
    else:
        print_results('Sentiment', sentiment, dataset, sentiment_prediction)


    opinion_prediction = predict_opinion(opinion, dataset)
    if opinion_prediction is None:
        print('Invalid opinion method')
        return 1
    else:
        print_results('Opinion Towards', opinion, dataset, opinion_prediction)


    stance_prediction = predict_stance(stance, dataset, sentiment_prediction, opinion_prediction)
    if stance_prediction is None:
        print('Invalid stance method')
        return 1
    else:
        print_results('Stance', stance, dataset, stance_prediction)

    dm.save(output, dataset, DELIMITER)
    return 0


def predict_sentiment(sentiment, dataset):
    if sentiment == 'wordnetaffect':
        return wa.predict(dataset['Tweet'])
    elif sentiment == 'sentiwordnet':
        return sw.predict(dataset['Tweet'])
    elif sentiment == 'apprentissage':
        dataset_train = dm.format(dm.load('./dataset/train.csv', ','))
        dataset_test = dm.format(dm.load('./dataset/test.csv', ','))
        return ap.predict(dataset_train['Tweet'],dataset_train['Sentiment'],dataset_test['Tweet'],dataset_test['Sentiment'])
    return None


def predict_opinion(opinion, dataset):
    if opinion == 'neural_network':
        dataset_train = dm.format(dm.load('./dataset/train.csv', ','))
        dataset_test = dm.format(dm.load('./dataset/test.csv', ','))
        model = nn.train(dataset_train, dataset_test)
        return nn.predict(model, dataset)
    return None


def predict_stance(stance, dataset, sentiment_prediction, opinion_prediction):
    if stance == 'stance':
        dataset_train = dm.format(dm.load('./dataset/train.csv', DELIMITER))
        dataset_test = dm.format(dm.load('./dataset/test.csv', DELIMITER))
        model, lb_target, lb_opinion, lb_sentiment, lb_stance = sd.get_model(dataset_train, dataset_test)
        return sd.predict_stance(dataset, model, opinion_prediction, sentiment_prediction, lb_target, lb_opinion, lb_sentiment, lb_stance)
    return None


def print_results(column, method, dataset, prediction):
    try:
        truth = dataset[column]
        success_rate = metrics.success_rate(truth, prediction)
        print('%s results: %s  ' % (method, success_rate))
    except:
        pass


def define_parameters(args):
    filename = './dataset/train.csv'
    sentiment = 'apprentissage'
    opinion = 'neural_network'
    stance = 'stance'
    output = './output.csv'

    for arg in args:
        if arg.startswith('--filename='):
            filename = arg.split('--filename=')[1]
        elif arg.startswith('--sentiment='):
            sentiment = arg.split('--sentiment=')[1]
        elif arg.startswith('--opinion='):
            opinion = arg.split('--opinion=')[1]
        elif arg.startswith('--stance='):
            stance = arg.split('--stance=')[1]
        elif arg.startswith('--output='):
            output = arg.split('--output=')[1]
    return filename, sentiment, opinion, stance, output


if __name__ == '__main__':
    exit(main())
