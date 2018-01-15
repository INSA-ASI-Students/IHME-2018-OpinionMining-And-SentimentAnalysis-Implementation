import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

FILENAME = 'dist/stance.model.pkl'


def export_model(model):
    joblib.dump(model, FILENAME)


def import_model():
    return joblib.load(FILENAME)


def encode_data(data, lb=LabelEncoder()):
    return lb.fit_transform(data)


def prepare_data(dataset_train, dataset_test):
    # Encode classes
    target_train = encode_data(dataset_train['Target'])
    opinion_train = encode_data(dataset_train['Opinion Towards'])
    sentiment_train = encode_data(dataset_train['Sentiment'])
    stance_train = encode_data(dataset_train['Stance'])

    target_test = encode_data(dataset_test['Target'])
    opinion_test = encode_data(dataset_test['Opinion Towards'])
    sentiment_test = encode_data(dataset_test['Sentiment'])
    stance_test = encode_data(dataset_test['Stance'])

    X_train = np.array([target_train, opinion_train, sentiment_train])
    X_train = np.transpose(X_train)
    X_test = np.array([target_test, opinion_test, sentiment_test])
    X_test = np.transpose(X_test)
    return X_train, X_test, stance_train, stance_test


def train(dataset_train, dataset_test):
    print('Learn SVM model for stance detection')
    X_train, X_test, y_train, y_test = prepare_data(dataset_train, dataset_test)
    model = SVC(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train, y_train)
    dec = model.predict(X_test)
    error = sum((y_test - dec) != 0) / len(y_test) * 100
    print('Test good rate in percent : ')
    print(100 - error)
    return model


def predict(dataset, model, lb=LabelEncoder()):
    data = np.array([
        encode_data(dataset['Target']),
        encode_data(dataset['Opinion Towards']),
        encode_data(dataset['Sentiment']
                    )])
    data = np.transpose(data)
    prediction = model.predict(data)
    return prediction.tolist()
