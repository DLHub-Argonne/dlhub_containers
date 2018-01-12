
import numpy as np
import json
import os
import pickle

from home_run.models import HRModel

from flask import Flask, request, jsonify
app = Flask(__name__)

from sklearn.externals import joblib

# Define where the models are
model_file = "./model/test.h5"

# Instantiate a HRModel for sklearn type
ml = HRModel(model_type="keras",
			 model_file=model_file)

@app.route("/", methods=["POST"])
def slash_post():
	body = request.get_json()

	if body['input']:
		data = body['input']
		print(np.array(data).shape)
		res = ml.model.predict(np.array(data))

		body.update({"output":res.tolist()})
		return json.dumps(body)

#Test route
@app.route("/test", methods=["GET"])
def slash_test():
	from keras.datasets import mnist

	body = {"input":[], "output":[]}
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	num_items = 2

	data = {
	    "batch":False,
	    "input":x_test[0:num_items].reshape(num_items, 28, 28, 1).tolist()
	}

	res = ml.predict(np.array(data['input'][0:num_items]).reshape(num_items,28,28,1))
	body.update({"output":res.tolist()})
	return json.dumps(body)
