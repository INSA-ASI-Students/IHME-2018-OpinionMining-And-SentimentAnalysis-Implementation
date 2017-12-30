from utils import dependancies
from utils import metrics
from utils import dataset_manager as dm

from sentiment_detection import sentiwordnet as sw
from sentiment_detection import wordnetaffect as wa


def main():
    # dependancies.download_nltk()
    train_dataset = dm.format(dm.load('./dataset/train.csv', ','))
    test_dataset = dm.format(dm.load('./dataset/test.csv', ','))
    prediction = wa.predict(train_dataset['Tweet'])
    error_rate = metrics.error_rate(test_dataset['Sentiment'], prediction)
    print(error_rate)
    return 0


if __name__ == '__main__':
    exit(main())
