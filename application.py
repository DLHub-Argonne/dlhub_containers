
import numpy as np
import json
import os
import pickle

from flask import Flask, request, jsonify
app = Flask(__name__)

from sklearn.externals import joblib 

def train():
	import numpy as np
	from sklearn.datasets import load_iris
	from sklearn.ensemble import RandomForestClassifier

	from sklearn import linear_model, datasets

	iris = load_iris()
	X = iris.data[:, :]  # we only take the first two features.
	y = iris.target

	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(X, y)
	print(clf.predict(X))
	joblib.dump(clf, 'filename.pkl')
	return clf

def load(model_file):
	clf = joblib.load(model_file)
	return clf

def predict(clf, data):
	X = np.array(data)
	print(X)
	return clf.predict(X)

#print("=== training ===")
#clf = train()

@app.route("/", methods=["POST"])
def slash_post():
	body = request.get_json()
	print(body)
	print("=== Loading ===")
	clf = load('./filename.pkl')

	if body['batch'] is True:
		print("=== batch predicting ===")
		data_file='.data/data.json'
		data = json.load(open(data_file))
		res = predict(clf, data=data)
		print(res)
		return json.dumps({"output":res.tolist()})
	elif body['input']:
		data = body['input']
		res = predict(clf, data=data)
		output = res.tolist()
		body.update({"output":output})
		return json.dumps(body)
