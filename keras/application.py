import json
from home_run.models import KerasModel

# Create the flask app
from flask import Flask, request
app = Flask(__name__)
app.config['DEBUG'] = False


# Load in options form disk
with open('options.json') as fp:
    options = json.load(fp)

# Instantiate a HRModel for sklearn type
ml = KerasModel(**options)

@app.route("/", methods=["POST"])
def slash_post():
    body = request.get_json()
    return json.dumps(ml.run(body))

if __name__ == "__main__":
    app.run(debug=False)
