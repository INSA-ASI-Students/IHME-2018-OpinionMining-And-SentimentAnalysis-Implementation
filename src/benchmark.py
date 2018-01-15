import csv
import gc

from learn_opinion import neural_network as nn
from utils import dataset_manager as dm

from keras.optimizers import Adam
from keras import backend as K

def benchmark_opinion(dataset_train, dataset_test, embedding_vector_length, batch_size, epochs, lr, beta_1, beta_2, decay):
    train_tweets, train_subjects = nn.format_dataset(dataset_train)
    train_labels = nn.format_labels(dataset_train)

    # Get Test tweets, subjects and labels
    test_tweets, test_subjects = nn.format_dataset(dataset_test)
    test_labels = nn.format_labels(dataset_test)

    # Train
    keras_model = nn.create_model(
        nn.MAX_SENTENCE_LENGTH,
        nn.MAX_SUBJECT_LENGTH,
        nn.VOCAB_SIZE,
        embedding_vector_length
    )

    optimizer = Adam(lr, beta_1, beta_2, decay=decay)
    keras_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    keras_model.fit(
        {'tweets': train_tweets, 'subjects': train_subjects},
        train_labels,
        batch_size=batch_size,
        epochs=epochs
    )

    # Evaluate
    train_score = keras_model.evaluate(
        {'tweets': train_tweets, 'subjects': train_subjects},
        train_labels
    )

    test_score = keras_model.evaluate(
        {'tweets': test_tweets, 'subjects': test_subjects},
        test_labels
    )
    del keras_model
    K.clear_session()

    return train_score, test_score

def main():
    lrs = [1, 0.1, 0.01]
    betas_1 = [0.9, 0.5, 0.1]
    betas_2 = [0.9, 0.5, 0.1]
    decays = [10, 1, 0.1, 0.01]
    epochs = [10]
    batch_sizes = [32, 64]
    embedding_vector_lengths = [32]

    dataset_train = dm.format(dm.load('./dataset/train.csv', ','))
    dataset_test = dm.format(dm.load('./dataset/test.csv', ','))

    with open('benchmark_opinion.csv', 'w') as bench_file:
        writer = csv.writer(bench_file)
        for lr in lrs:
            for beta_1 in betas_1:
                for beta_2 in betas_2:
                    for decay in decays:
                        for epoch in epochs:
                            for batch_size in batch_sizes:
                                for embedding_vector_length in embedding_vector_lengths:
                                    row = [
                                        lr,
                                        beta_1,
                                        beta_2,
                                        decay,
                                        epoch,
                                        batch_size,
                                        embedding_vector_length
                                    ]
                                    print('START :', row)
                                    train_score, test_score = benchmark_opinion(
                                        dataset_train,
                                        dataset_test,
                                        embedding_vector_length,
                                        batch_size,
                                        epoch,
                                        lr,
                                        beta_1,
                                        beta_2,
                                        decay
                                    )
                                    gc.collect()
                                    row.append(train_score)
                                    row.append(test_score)
                                    print('DONE :', row)
                                    writer.writerow(
                                        row
                                    )

    return 0

if __name__ == '__main__':
    exit(main())