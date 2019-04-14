import json

import pandas as pd
from flask import Response, request, jsonify

from src.core.analytics_service import AnalyticsService
from src.core.prediction_service import PredictionService
from src.run import app

analytics_service = AnalyticsService()


@app.route("/")
def hello():
    all_surveys = analytics_service.get_all_surveys()
    print(all_surveys)

    return Response(all_surveys, mimetype='application/json')


@app.route("/submission-batch", methods=['POST'])
def post_submission_batch():
    submission_batch = request.get_json()
    return jsonify(submission_batch)


