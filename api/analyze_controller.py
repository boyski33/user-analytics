from run import app


@app.route("/")
def hello():
    return "working"
