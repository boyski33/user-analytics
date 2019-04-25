import py_eureka_client.eureka_client as eureka_client
from flask import Flask

from config import config

app = Flask(__name__)

# import not at the top because app definition is needed
from api.analytics_controller import *


def run_app():

    eureka_client.init(eureka_server=config["eureka_server_url"],
                       app_name=config["service_name"],
                       instance_port=config["service_port"])

    try:
        res = eureka_client.do_service("HIPPO-CORE-SURVEY", "/surveys")
        print("result: " + res)
        print("{}SUCCESSFULLY REGISTERED ON EUREKA SERVER{}".format("\033[92m", "\033[0m"))
    except Exception as e:
        print(e)

    app.run(port=config["service_port"])


if __name__ == '__main__':
    run_app()
