import requests

# Pickled inputs from `util` container
data = [{'composition': 'Al2O3', 'composition_object': 'gANjcHltYXRnZW4uY29yZS5jb21wb3NpdGlvbgpDb21wb3NpdGlvbgpxACmBcQF9cQIoWA4AAABh\nbGxvd19uZWdhdGl2ZXEDiVgHAAAAX25hdG9tc3EER0AUAAAAAAAAWAUAAABfZGF0YXEFfXEGKGNw\neW1hdGdlbi5jb3JlLnBlcmlvZGljX3RhYmxlCkVsZW1lbnQKcQdYAgAAAEFscQiFcQlScQpHQAAA\nAAAAAABoB1gBAAAAT3ELhXEMUnENR0AIAAAAAAAAdXViLg==\n'},
        {'composition': 'NaCl', 'composition_object': 'gANjcHltYXRnZW4uY29yZS5jb21wb3NpdGlvbgpDb21wb3NpdGlvbgpxACmBcQF9cQIoWA4AAABh\nbGxvd19uZWdhdGl2ZXEDiVgHAAAAX25hdG9tc3EER0AAAAAAAAAAWAUAAABfZGF0YXEFfXEGKGNw\neW1hdGdlbi5jb3JlLnBlcmlvZGljX3RhYmxlCkVsZW1lbnQKcQdYAgAAAE5hcQiFcQlScQpHP/AA\nAAAAAABoB1gCAAAAQ2xxC4VxDFJxDUc/8AAAAAAAAHV1Yi4=\n'}]
result = requests.post("http://localhost:5000/", json=data)
print(result.status_code)
print(result.json())

