import requests
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_items = 2

print(x_test.shape)

data = [
 {'features':item.reshape(28,28,1).tolist()} for item in x_test[0:num_items]
]


result = requests.post("http://0.0.0.0:5000/", json=data)
print(result.status_code)
print(result.json())
