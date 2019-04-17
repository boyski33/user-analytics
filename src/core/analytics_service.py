import json

import pandas as pd

from src.core.prediction_service import PredictionService

mock_feature_cols = ['Q1', 'Q2', 'Q3']


class AnalyticsService:

    def __init__(self):
        self.prediction_service = PredictionService()

    def train(self, survey_id: str, data: dict):
        df = self.convert_json_to_panda(data)
        self.prediction_service.train_and_persist_model(survey_id, df, mock_feature_cols)

    def predict(self, survey_id, data: dict):
        example = self.normalize_example(data)

        age, gender = self.prediction_service.predict_age_and_gender(survey_id, example)

        return list([int(a) for a in age]), list(gender)

    @staticmethod
    def convert_json_to_panda(data: dict) -> pd.DataFrame:
        print(data)

        file = 'mock_data/test_data.json'
        with open(file) as f:
            json_data = json.load(f)

        return pd.DataFrame.from_dict(
            data=json_data,
            orient='columns'
        )

    @staticmethod
    def normalize_example(data: dict) -> list:
        return [['red', 'history', 'basketball']]
