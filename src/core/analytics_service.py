from typing import List

import pandas as pd

from src.config import config
from src.core.prediction_service import PredictionService
from src.model.submission import Submission

class_columns = config['class_columns']

gender_map = {
    'male': 'm',
    'female': 'f'
}


class AnalyticsService:

    def __init__(self):
        self.prediction_service = PredictionService()

    def train(self, survey_id: str, dataset: list):
        df = self.convert_dataset_to_panda(dataset)
        feature_cols = list(df.drop(columns=class_columns).columns.values)

        self.prediction_service.train_and_persist_model(survey_id, df, feature_cols)

    def predict(self, survey_id, data: dict) -> List[Submission]:
        submissions: List[Submission] = data['submissions']
        examples = self.normalize_examples(submissions)

        age, gender = self.prediction_service.predict_age_and_gender(survey_id, examples)

        for i in range(len(age)):
            # todo refactor
            submissions[i].user.age = age[i]
            submissions[i].user.gender = gender[i]

        return submissions

    @staticmethod
    def convert_dataset_to_panda(dataset: list) -> pd.DataFrame:
        normalized = AnalyticsService.normalize_dataset(dataset)

        return pd.DataFrame.from_dict(
            data=normalized,
            orient='columns'
        )

    @staticmethod
    def normalize_dataset(dataset: list):
        normalized = []
        for d in dataset:
            normalized.append(AnalyticsService.normalize_data(d))

        return normalized

    @staticmethod
    def normalize_data(data: dict) -> dict:
        result = {
            'age': data['userAge'],
            'gender': gender_map[data['userGender']]
        }

        answers = AnalyticsService.normalize_answers(data['answeredQuestions'])
        result = {**result, **answers}

        return result

    @staticmethod
    def normalize_answers(answers: list) -> dict:
        normalized = {}
        for ans in answers:
            normalized[ans['question']['key']] = str(ans['answer']).lower()

        return normalized

    @staticmethod
    def normalize_examples(submissions: List[Submission]) -> list:
        result = []

        for sub in submissions:
            answers = []
            for ans in sub['answers']:
                answers.append(str(ans['answer']).lower())
            result.append(answers)

        return result
