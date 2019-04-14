import json
import os

import pandas as pd
from pymongo import MongoClient

from src.core.prediction_service import PredictionService


class AnalyticsService:

    def __init__(self):
        self.client = MongoClient(
            'mongodb+srv://admin:admin@hippo-cluster-gya0k.mongodb.net/hippo-survey-db?retryWrites=true')
        self.db = self.client['hippo-survey-db']

    def get_all_surveys(self):
        users = self.db['users']
        cursor = users.find({})
        result = [x for x in cursor]

        return str(result)

    def mock_data(self):
        file = 'mock_data/test_data.json'
        with open(file) as f:
            json_data = json.load(f)

        return pd.DataFrame.from_dict(
            data=json_data,
            orient='columns'
        )

    def test(self):
        df = self.mock_data()

        prediction_service = PredictionService(df, feature_columns=('Q1', 'Q2', 'Q3'))
        prediction_service.train()