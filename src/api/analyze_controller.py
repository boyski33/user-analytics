from src.run import app
from src.core.idk_service import Service
from flask import Response
import json

service = Service()


@app.route("/")
def hello():
    all_surveys = service.get_all_surveys()
    print(all_surveys)

    return Response(all_surveys, mimetype='application/json')
