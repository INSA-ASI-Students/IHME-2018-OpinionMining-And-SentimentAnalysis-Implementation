import dataset_toolbox as dp


def main():
    dp.init()
    train_dataset = dp.format_dataset(dp.load_dataset('./StanceDataset/train.csv', ','))
    test_dataset = dp.format_dataset(dp.load_dataset('./StanceDataset/test.csv', ','))
    return 0


if __name__ == '__main__':
    exit(main())
