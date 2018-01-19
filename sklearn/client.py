import requests

data = [
    {'features': [1,2,3,4]},
    {'features': [1,2,3,5]},
]
result = requests.post("http://localhost:5000/", json=data)
print(result.status_code)
print(result.json())

