import json

import pandas as pd

from src.core.prediction_service import PredictionService

mock_feature_cols = ['Q1', 'Q2', 'Q3']

class AnalyticsService:

    def __init__(self):
        self.prediction_service = PredictionService()

    def mock_data(self):
        file = 'mock_data/test_data.json'
        with open(file) as f:
            json_data = json.load(f)

        return pd.DataFrame.from_dict(
            data=json_data,
            orient='columns'
        )

    def train(self):
        df = self.mock_data()
        self.prediction_service.train_and_persist_model(df, mock_feature_cols)

    def predict(self, survey_id, example):
        age, gender = self.prediction_service.predict_age_and_gender(survey_id, example)

        return list([int(a) for a in age]), list(gender)