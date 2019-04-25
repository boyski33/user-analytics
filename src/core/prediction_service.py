import pickle

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model.base import LinearModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

from src.config import config

class_columns = config['class_columns']
col_age = class_columns[:1]
col_gender = class_columns[1:]


class PredictionService:

    def __init__(self):
        self.client = MongoClient(config['mongo_connection'])
        self.db = self.client['hippo-survey-db']
        self.models = self.db['ml-models']

    def train_and_persist_model(self, survey_id: str, df: pd.DataFrame, feature_columns: list):
        # Dummy Variables & One Hot Encoding
        feature_sets = pd.get_dummies(df, columns=feature_columns).drop(columns=class_columns).values
        genders = df[col_gender].values
        ages = df[col_age].values

        encoder = OneHotEncoder(handle_unknown='ignore').fit(df[feature_columns].values)

        age_model = self.linear_regression_train(feature_sets, ages)
        gender_model = self.random_forest_train(feature_sets, genders)

        self.persist_model(survey_id, age_model, gender_model, encoder)

    def persist_model(self, survey_id, age_model, gender_model, encoder):
        age_model_binary = pickle.dumps(age_model)
        gender_model_binary = pickle.dumps(gender_model)
        encoder_binary = pickle.dumps(encoder)

        self.models.update_one(
            filter={'surveyId': survey_id},
            update={'$set': {
                'surveyId': survey_id,
                'ageModel': age_model_binary,
                'genderModel': gender_model_binary,
                'encoder': encoder_binary
            }},
            upsert=True
        )

    def predict_age_and_gender(self, survey_id: str, examples: list):
        data = self.models.find_one({'surveyId': survey_id})

        if data is None:
            return [], []

        age_model: LinearModel = pickle.loads(data['ageModel'])
        gender_model = pickle.loads(data['genderModel'])
        encoder: OneHotEncoder = pickle.loads(data['encoder'])

        examples = self.one_hot_encode(encoder, examples)

        age: np.ndarray = age_model.predict(examples)
        gender: np.ndarray = gender_model.predict(examples)

        return age.tolist(), gender.tolist()

    ### TRAINING METHODS ###

    @staticmethod
    def linear_regression_train(x, y) -> LinearRegression:
        lr_model = LinearRegression()
        return lr_model.fit(x, y.ravel())

    @staticmethod
    def gaussian_nb_train(x, y) -> GaussianNB:
        nb_model = GaussianNB()
        nb_model.fit(x, y.ravel())
        return nb_model

    @staticmethod
    def random_forest_train(x, y) -> RandomForestClassifier:
        rf_model = RandomForestClassifier(class_weight='balanced', random_state=11)
        return rf_model.fit(x, y.ravel())

    @staticmethod
    def logistic_regression_train(x, y) -> LogisticRegression:
        lr_model = LogisticRegression(C=2.2, class_weight='balanced')
        lr_model.fit(x, y.ravel())
        return lr_model

    @staticmethod
    def one_hot_encode(encoder: OneHotEncoder, example: list):
        np_arr = np.array(example)
        if np_arr.ndim == 1:
            example = [example]

        return encoder.transform(example).toarray()

    ### TESTING METHODS ###

    def age_model_test(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=31)
        linreg_model = self.linear_regression_train(X_train, y_train)

        print("Age mean error: {0:.4f}".format(metrics.mean_absolute_error(y_test, linreg_model.predict(X_test))))

    def gender_model_test(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=31)
        nb_model = self.gaussian_nb_train(X_train, y_train)

        print("Gender accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_model.predict(X_test))))
