import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

FILENAME = 'dist/stance.model.pkl'


def export_model(model):
    joblib.dump(model, FILENAME)


def import_model():
    return joblib.load(FILENAME)


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


def get_model(dataset_train, dataset_test):
    [lb_target, lb_opinion, lb_sentiment, lb_stance] = [LabelEncoder(
    ), LabelEncoder(), LabelEncoder(), LabelEncoder()]
    X_train, X_test, y_train, y_test = prepare_data(
        dataset_train, dataset_test, lb_target, lb_opinion, lb_sentiment, lb_stance)
    print('Learn SVM model for stance detection')
    model = SVC(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train, y_train)
    dec = model.predict(X_test)
    error = sum((y_test - dec) != 0) / len(y_test) * 100
    print('Test good rate in percent : ')
    print(100 - error)
    return model, lb_target, lb_opinion, lb_sentiment, lb_stance


def predict_stance(dataset, model, opinion_prediction, sentiment_prediction, lb_target, lb_opinion, lb_sentiment, lb_stance):
    data = np.array([encode_data(lb_target, dataset['Target']), encode_data(
        lb_opinion, dataset['Opinion Towards']), encode_data(lb_sentiment, sentiment_prediction)])
    data = np.transpose(data)
    prediction = model.predict(data)
    return lb_stance.inverse_transform(prediction)
