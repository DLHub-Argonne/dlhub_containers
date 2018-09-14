import dlhub_shim

data = [
    {'features': [1,2,3,4]},
    {'features': [1,2,3,5]},
]
result = dlhub_shim.run(data)
print(result)

