import requests
from keras.datasets import mnist

DATASET = "candle"


if DATASET == "mnist": 
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	num_items = 2

	data = [
	 {'features':item.reshape(28,28,1).tolist()} for item in x_test[0:num_items]
	]
elif DATASET == "candle":
	import pandas as pd

	df = pd.read_csv('./data/x_test_candle.csv')

	data = [
	    {'features':row.values.tolist()} for index, row in df.iterrows()
	]
else:
	data = []

result = requests.post("http://0.0.0.0:5000/", json=data)
res = result.json()
print(result.status_code)
print([item['prediction'] for item in res])