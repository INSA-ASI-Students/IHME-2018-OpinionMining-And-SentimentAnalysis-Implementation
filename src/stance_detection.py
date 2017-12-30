from utils import dependancies
from utils import metrics
from utils import dataset_manager as dm

from sentiment_detection import sentiwordnet as sw
from sentiment_detection import wordnetaffect as wa


def main():
    # dependancies.download_nltk()
    train_dataset = dm.format(dm.load('./dataset/train.csv', ','))
    test_dataset = dm.format(dm.load('./dataset/test.csv', ','))

    wa_prediction = wa.predict(train_dataset['Tweet'])
    wa_success_rate = metrics.success_rate(test_dataset['Sentiment'], wa_prediction)

    sw_prediction = sw.predict(train_dataset['Tweet'])
    sw_success_rate = metrics.success_rate(test_dataset['Sentiment'], sw_prediction)

    print('wordnetaffect results: %s  ' % (wa_success_rate))
    print('sentiwordnet results: %s  ' % (sw_success_rate))

    return 0


if __name__ == '__main__':
    exit(main())
