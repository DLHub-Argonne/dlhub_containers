import requests

data = [
    {'composition': 'Al2O3'},
    {'composition': 'NaCl'},
]
result = requests.post("http://localhost:5000/", json=data)
print(result.status_code)
print(result.json())

