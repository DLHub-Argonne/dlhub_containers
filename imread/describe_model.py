from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from dlhub_sdk.utils.schemas import validate_against_dlhub_schema
from PIL import Image
import json

# Create a model that performs the 'no-opt' operation
model = PythonStaticMethodModel.from_function_pointer(Image.open)

model.set_title("Image reading function")
model.set_name('read_image')
model.set_abstract("Reads an image file into an aray")
model.set_inputs('file', 'Image file', file_type='image/*')
model.set_outputs('ndarray', 'Image contents', shape=[None, None, None])

# Add PIL as a dependency
model.add_requirement('PIL')

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
validate_against_dlhub_schema(metadata, 'servable')
print(json.dumps(metadata, indent=2))
with open('dlhub.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)
