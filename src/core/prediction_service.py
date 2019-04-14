import json

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

class_columns = ['age', 'gender']
col_age = class_columns[:1]
col_gender = class_columns[1:]


class PredictionService:

    def __init__(self, df: pd.DataFrame, feature_columns=None):
        self.df = df

        if feature_columns is None:
            feature_columns = ['Q1', 'Q2', 'Q3']

        self.feature_columns = feature_columns
        self.age_model = LinearRegression()
        self.gender_model = LogisticRegression()

    def train(self):

        # Dummy Variables & One Hot Encoding
        feature_sets = pd.get_dummies(self.df, columns=self.feature_columns).drop(columns=class_columns).values
        genders = self.df[col_gender].values
        ages = self.df[col_age].values

        enc = OneHotEncoder()
        enc.fit(self.df[self.feature_columns].values)

        entry = [['blue', 'history', 'tennis']]

        encoded_entry = self.one_hot_encode(enc, entry)

        self.age_model = self.linear_regression_train(feature_sets, ages)
        self.gender_model = self.random_forest_train(feature_sets, genders)

        predicted_age = self.age_model.predict(encoded_entry)
        predicted_gender = self.gender_model.predict(encoded_entry)

        print([int(age) for age in predicted_age])
        print(predicted_gender)

        # self.gaussian_nb_test(feature_sets, genders)
        # self.random_forest_test(feature_sets, genders)
        # self.logistic_regression_test(feature_sets, genders)
        # self.linear_regression_test(feature_sets, ages)

    ### TRAINING METHODS ###

    def linear_regression_train(self, x, y) -> LinearRegression:
        lr_model = LinearRegression()
        return lr_model.fit(x, y.ravel())

    def gaussian_nb_train(self, x, y) -> GaussianNB:
        nb_model = GaussianNB()
        nb_model.fit(x, y.ravel())
        return nb_model

    def random_forest_train(self, x, y) -> RandomForestClassifier:
        rf_model = RandomForestClassifier(class_weight='balanced', random_state=11)
        return rf_model.fit(x, y.ravel())

    def logistic_regression_train(self, x, y) -> LogisticRegression:
        lr_model = LogisticRegression(C=2.2, class_weight='balanced')
        lr_model.fit(x, y.ravel())
        return lr_model

    ### TESTING METHODS ###

    def age_model_test(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=31)
        linreg_model = self.linear_regression_train(X_train, y_train)

        print("Age mean error: {0:.4f}".format(metrics.mean_absolute_error(y_test, linreg_model.predict(X_test))))

    def gender_model_test(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
        nb_model = self.gaussian_nb_train(X_train, y_train)

        print("Gender accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_model.predict(X_test))))

    def one_hot_encode(self, encoder, row):
        np_arr = np.array(row)
        if np_arr.ndim == 1:
            row = [row]

        return encoder.transform(row).toarray()

