from flask import Response, request, jsonify

from src.core.analytics_service import AnalyticsService
from src.run import app

analytics_service = AnalyticsService()


@app.route("/submission-batch", methods=['POST'])
def post_submission_batch():
    submission_batch = request.get_json()
    return jsonify(submission_batch)


@app.route("/train", methods=['POST'])
def train_model():
    survey_data = request.get_json()

    analytics_service.train(survey_data['surveyId'], survey_data['submissions'])

    return Response()


@app.route("/predict", methods=['POST'])
def predict():
    age, gender = analytics_service.predict('dummy_ID', {})

    return jsonify(age=age, gender=gender)
