import json

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder


def train():
    df = get_data()

    feature_col_names = ['Q1', 'Q2', 'Q3']
    class_columns = ['age', 'gender']
    col_age = class_columns[:1]
    col_gender = class_columns[1:]

    # Dummy Variables & One Hot Encoding
    feature_sets = pd.get_dummies(df, columns=feature_col_names).drop(columns=class_columns).values
    genders = df[col_gender].values
    ages = df[col_age].values

    enc = OneHotEncoder()
    enc.fit(df[feature_col_names].values)

    entry = [['blue', 'history', 'tennis']]

    encoded_entry = one_hot_encode(enc, entry)

    lr_model = linear_regression_train(feature_sets, ages)
    rf_model = random_forest_train(feature_sets, genders)

    predicted_age = lr_model.predict(encoded_entry)
    predicted_gender = rf_model.predict(encoded_entry)

    print([int(age) for age in predicted_age])
    print(predicted_gender)

    # gaussian_nb_test(feature_sets, genders)
    # random_forest_test(feature_sets, genders)
    # logistic_regression_test(feature_sets, genders)


### TRAINING METHODS ###

def linear_regression_train(x, y) -> LinearRegression:
    lr_model = LinearRegression()
    return lr_model.fit(x, y.ravel())


def gaussian_nb_train(x, y) -> GaussianNB:
    nb_model = GaussianNB()
    nb_model.fit(x, y.ravel())
    return nb_model


def random_forest_train(x, y) -> RandomForestClassifier:
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=11)
    return rf_model.fit(x, y.ravel())

def logistic_regression_train(x, y) -> LogisticRegression:
    lr_model = LogisticRegression(C=2.2, class_weight='balanced')
    lr_model.fit(x, y.ravel())
    return lr_model


### TESTING METHODS ###

def gaussian_nb_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
    nb_model = gaussian_nb_train(X_train, y_train)

    print("NB accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_model.predict(X_test))))


def random_forest_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
    rf_model = random_forest_train(X_train, y_train)

    print("RF accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_model.predict(X_test))))


def logistic_regression_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
    logreg_model = logistic_regression_train(X_train, y_train)

    print("LogReg accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, logreg_model.predict(X_test))))


def one_hot_encode(encoder, row):
    np_arr = np.array(row)
    if np_arr.ndim == 1:
        row = [row]

    return encoder.transform(row).toarray()


def get_data():
    file = './test_data.json'
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
