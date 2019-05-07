#!/usr/bin/env python

from flask import Flask

from config import config

app = Flask(__name__)

# import not at the top because app definition is needed
from api.analytics_controller import *


def run_app():
    app.run(port=config["service_port"])


if __name__ == '__main__':
    run_app()
