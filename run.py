from flask import Flask
from config import config
import requests
import py_eureka_client.eureka_client as eureka_client

app = Flask(__name__)

# import not at the top because app definition is needed
from api.analyze_controller import *


def run_app():
    eureka_client.init(eureka_server=config["eureka_server_url"],
                       app_name=config["service_name"],
                       instance_port=config["service_port"])

    try:
        res = eureka_client.do_service("HIPPO-DOMAIN-SURVEY", "/surveys")
        print("result: " + res)
    except requests.HTTPError as e:
        print(e)

    app.run(port=config["service_port"])


if __name__ == '__main__':
    run_app()
