import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def encode_data(lb, data):
    return lb.fit_transform(data)

def fusion_data(dataset_train, dataset_test, label):
    return np.hstack([np.array(dataset_train[label]), np.array(dataset_test[label])])

def prepare_data(dataset_train, dataset_test, lb_target, lb_opinion, lb_sentiment, lb_stance):
    # Data fusion
    target = fusion_data(dataset_train, dataset_test, 'Target')
    opinion = fusion_data(dataset_train, dataset_test, 'Opinion Towards')
    sentiment = fusion_data(dataset_train, dataset_test, 'Sentiment')
    stance = fusion_data(dataset_train, dataset_test, 'Stance')

    # Encode classes
    target_label = encode_data(lb_target, target)
    opinion_label = encode_data(lb_opinion, opinion)
    sentiment_label = encode_data(lb_sentiment, sentiment)
    stance_label = encode_data(lb_stance, stance)

    # Split data
    X = np.array([target_label, opinion_label, sentiment_label])
    X = np.transpose(X)
    return train_test_split(X, stance_label, test_size=0.15, random_state=1)

def get_model(dataset_train, dataset_test):
    [lb_target, lb_opinion, lb_sentiment, lb_stance] = [preprocessing.LabelEncoder(), preprocessing.LabelEncoder(), preprocessing.LabelEncoder(), preprocessing.LabelEncoder()]
    X_train, X_test, y_train, y_test = prepare_data(dataset_train, dataset_test, lb_target, lb_opinion, lb_sentiment, lb_stance)
    print('Learn SVM model for stance detection')
    model = SVC(kernel='rbf', C=100,gamma=0.1)
    model.fit(X_train, y_train)
    dec = model.predict(X_test)
    error = sum((y_test - dec) != 0)/len(y_test)*100
    print('Test good rate in percent : ')
    print(100-error)
    return model, lb_target, lb_opinion, lb_sentiment, lb_stance

def predict_stance(dataset, model, opinion_prediction, sentiment_prediction, lb_target, lb_opinion, lb_sentiment, lb_stance):
    data = np.array([encode_data(lb_target, dataset['Target']), encode_data(lb_opinion, dataset['Opinion Towards']), encode_data(lb_sentiment, sentiment_prediction)])
    data = np.transpose(data)
    prediction = model.predict(data)
    return lb_stance.inverse_transform(prediction)
