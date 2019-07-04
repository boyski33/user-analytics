from datetime import date

import pandas as pd

from config import config
from prediction_service import PredictionService

class_columns = config['class_columns']
supported_types = config['supported_question_types']


class AnalyticsService:

    def __init__(self):
        self.prediction_service = PredictionService()

    def train(self, survey_id: str, dataset: list):
        df = self.convert_dataset_to_panda(dataset)
        feature_cols = list(df.drop(columns=class_columns).columns.values)

        if len(feature_cols) > 0:
            self.prediction_service.train_and_persist_model(survey_id, df, feature_cols)

    def predict(self, survey_id, data: dict) -> list:
        submissions: list = data['submissions']
        examples = self.normalize_examples(submissions)

        age_list, gender_list = self.prediction_service.predict_age_and_gender(survey_id, examples)

        for i in range(len(age_list)):
            submissions[i]['user'] = {
                'dateOfBirth': AnalyticsService.convert_age_to_dob(int(age_list[i])),
                'gender': gender_list[i],
                'is_predicted': True
            }

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
            'gender': data['userGender']
        }

        answers = AnalyticsService.normalize_answers(data['answers'])
        result = {**result, **answers}

        return result

    @staticmethod
    def normalize_answers(answers: list) -> dict:
        normalized = {}
        for ans in answers:
            if ans['question']['controlType'] in supported_types:
                normalized[ans['question']['key']] = str(ans['answer']).lower()

        return normalized

    @staticmethod
    def normalize_examples(submissions: list) -> list:
        result = []

        for sub in submissions:
            answers = []
            for ans in sub['answers']:
                if ans['question']['controlType'] in supported_types:
                    answers.append(str(ans['answer']).lower())

            result.append(answers)

        return result

    @staticmethod
    def convert_age_to_dob(age: int):
        d = date.today()
        return d.replace(year=d.year - age)
