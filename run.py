from flask import Flask
import requests
import py_eureka_client.eureka_client as eureka_client

app = Flask(__name__)

service_name = "HIPPO-USER-ANALYTICS"
service_port = 8855


from api.analyze_controller import *


def run_app():
    eureka_client.init(eureka_server="http://localhost:8761/eureka", app_name=service_name, instance_port=service_port)

    try:
        res = eureka_client.do_service("HIPPO-DOMAIN-SURVEY", "/surveys")
        print("result: " + res)
    except requests.HTTPError as e:
        print(e)

    app.run(port=service_port)


if __name__ == '__main__':
    run_app()
