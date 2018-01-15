import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

FILENAME = 'dist/stance.model.pkl'
FILENAME_LABELS = 'dist/stance.labels.pkl'


def export_model(model):
    joblib.dump(model, FILENAME)


def import_model():
    return joblib.load(FILENAME)


def export_labels(labels):
    joblib.dump(labels, FILENAME_LABELS)


def import_labels():
    return joblib.load(FILENAME_LABELS)


def encode_data(lb, data):
    return lb.fit_transform(data)


def prepare_data(dataset_train, dataset_test, lb_target, lb_opinion, lb_sentiment, lb_stance):
    # Encode classes
    target_train = encode_data(lb_target, dataset_train['Target'])
    opinion_train = encode_data(lb_opinion, dataset_train['Opinion Towards'])
    sentiment_train = encode_data(lb_sentiment, dataset_train['Sentiment'])
    stance_train = encode_data(lb_stance, dataset_train['Stance'])

    target_test = encode_data(lb_target, dataset_test['Target'])
    opinion_test = encode_data(lb_opinion, dataset_test['Opinion Towards'])
    sentiment_test = encode_data(lb_sentiment, dataset_test['Sentiment'])
    stance_test = encode_data(lb_stance, dataset_test['Stance'])

    X_train = np.array([target_train, opinion_train, sentiment_train])
    X_train = np.transpose(X_train)
    X_test = np.array([target_test, opinion_test, sentiment_test])
    X_test = np.transpose(X_test)
    return X_train, X_test, stance_train, stance_test


def train(dataset_train, dataset_test):
    [lb_target, lb_opinion, lb_sentiment, lb_stance] = [LabelEncoder(
    ), LabelEncoder(), LabelEncoder(), LabelEncoder()]
    X_train, X_test, y_train, y_test = prepare_data(
        dataset_train, dataset_test, lb_target, lb_opinion, lb_sentiment, lb_stance)
    print('Learn SVM model for stance detection')
    model = SVC(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train, y_train)
    export_labels([lb_target, lb_opinion, lb_sentiment, lb_stance])
    return model


def predict(dataset, model):
    [lb_target, lb_opinion, lb_sentiment, lb_stance] = import_labels()
    data = np.array([
        encode_data(lb_target, dataset['Target']),
        encode_data(lb_opinion, dataset['Opinion Towards']),
        encode_data(lb_sentiment, dataset['Sentiment'])
    ])
    data = np.transpose(data)
    prediction = model.predict(data)
    return lb_stance.inverse_transform(prediction)
