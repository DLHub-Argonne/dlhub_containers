from dlhub_toolbox.models.servables.python import PythonStaticMethodModel
from dlhub_toolbox.utils.schemas import validate_against_dlhub_schema
import json

# Create a model that performs the 'no-opt' operation
model = PythonStaticMethodModel('app', 'noop')

model.set_title("No-op Function")
model.set_name('no-op')
model.set_abstract("A servable that returns whatever it was given as input")
model.set_inputs('python object', 'Anything', python_type='object')
model.set_outputs('python object', 'Same as the input', python_type='object')

# Add app.py, which holds the noop function, to the list of associated files to be submitted
model.add_file("app.py")

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
validate_against_dlhub_schema(metadata, 'servable')
print(json.dumps(metadata, indent=2))
with open('model_metadata.json', 'w') as fp:
    json.dump(metadata, fp)
