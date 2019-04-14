import json

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder


def train():
    df = get_stats()

    feature_col_names = ['Q1', 'Q2', 'Q3']
    class_col_name = ['gender']

    # Dummy Variables & One Hot Encoding
    x = pd.get_dummies(df, columns=feature_col_names).drop(columns=class_col_name).values
    y = df[class_col_name].values

    enc = OneHotEncoder()
    enc.fit(df[feature_col_names].values)

    entry = [['blue', 'history', 'basketball']]

    result = one_hot_encode(enc, entry)

    rf_model = random_forest_train(x, y)

    predictions = rf_model.predict(result)

    print(predictions)

    # gaussian_nb_test(x, y)
    # random_forest_test(x, y)
    # logistic_regression_test(x, y)


def random_forest_train(x, y) -> RandomForestClassifier:
    rf_model = RandomForestClassifier(class_weight='balanced')
    return rf_model.fit(x, y.ravel())


def gaussian_nb_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train.ravel())

    nb_predict_train = nb_model.predict(X_train)
    nb_predict_test = nb_model.predict(X_test)

    print("NB accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))


def random_forest_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(X_train, y_train.ravel())

    rf_predict_test = rf_model.predict(X_test)

    print("RF accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))


def logistic_regression_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
    logreg_model = LogisticRegression(C=2.2, class_weight='balanced')
    logreg_model.fit(X_train, y_train.ravel())

    logreg_predict_test = logreg_model.predict(X_test)

    print("Logistic regression accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, logreg_predict_test)))


def one_hot_encode(encoder, row):
    np_arr = np.array(row)
    if np_arr.ndim == 1:
        row = [row]

    return encoder.transform(row).toarray()


def get_stats():
    file = './test_data_2.json'
    with open(file) as f:
        json_data = json.load(f)

    return pd.DataFrame.from_dict(
        data=json_data,
        orient='columns'
    )


def main():
    train()


if __name__ == '__main__':
    main()
