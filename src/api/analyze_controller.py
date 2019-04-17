from flask import Response, request, jsonify

from src.core.analytics_service import AnalyticsService
from src.run import app

analytics_service = AnalyticsService()


@app.route("/submission-batch", methods=['POST'])
def post_submission_batch():
    submission_batch = request.get_json()
    return jsonify(submission_batch)


@app.route("/train/<survey_id>", methods=['PUT'])
def train_model(survey_id):
    analytics_service.train(survey_id, request.get_json())

    return Response()


@app.route("/predict", methods=['POST'])
def predict():
    age, gender = analytics_service.predict('dummy_ID', {})

    return jsonify(age=age, gender=gender)
