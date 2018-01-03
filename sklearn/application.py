
import numpy as np
import json
import os
import pickle

from homerun.models import MLModel

from flask import Flask, request, jsonify
app = Flask(__name__)

from sklearn.externals import joblib

# Define where the models are
model_file = "./model/iris.pkl"

# Instantiate a HRModel for sklearn type
ml = MLModel(model_type="sklearn",
             model_file=model_file)

@app.route("/", methods=["POST"])
def slash_post():
	body = request.get_json()

	if body['input']:
		data = body['input']
		res = ml.model.predict(np.array(data))
		body.update({"output":res.tolist()})
		return json.dumps(body)
