from flask import Response, request, jsonify

from analytics_service import AnalyticsService
from run import app

analytics_service = AnalyticsService()


@app.route("/train", methods=['POST'])
def train_model():
    survey_data = request.get_json()

    analytics_service.train(survey_data['surveyId'], survey_data['submissions'])

    return Response()


@app.route("/predict/<survey_id>", methods=['POST'])
def predict(survey_id: str):
    data = request.get_json()
    predicted_submissions: list = analytics_service.predict(survey_id, data)

    return jsonify({'submissions': predicted_submissions})
