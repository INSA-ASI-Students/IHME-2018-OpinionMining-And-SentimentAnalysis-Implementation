import dataset_toolbox as dp
from sentiment_detection import sentiwordnet as sw
from sentiment_detection import wordnetaffect as wa


def main():
    dp.init()
    train_dataset = dp.format_dataset(dp.load_dataset('./StanceDataset/train.csv', ','))
    test_dataset = dp.format_dataset(dp.load_dataset('./StanceDataset/test.csv', ','))
    prediction = wa.predict(test_dataset['Tweet'])
    error_rate = wa.check_performances(test_dataset['Sentiment'], prediction)
    print(error_rate)
    return 0


if __name__ == '__main__':
    exit(main())
