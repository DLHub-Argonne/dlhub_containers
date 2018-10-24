from dlhub_toolbox.models.servables.python import PythonStaticMethodModel
from dlhub_toolbox.utils.schemas import validate_against_dlhub_schema
import json
import os

# Create a model that invokes the "run" function from the
model = PythonStaticMethodModel.create_model('application', 'run')

#  Describe the inputs and outputs
model.set_inputs('list', 'Paths to all images in a dataset', item_type='string')
model.set_outputs('ndarray', 'Accumulated result of decoding all the images', shape=[208, 208])

#  Add provenance information
model.set_title("Deep-Learning Super-resolution Image Reconstruction (DSIR)")
model.set_name('dsir')
model.set_authors(['Duarte, Alex'], ['The Institute of Photonic Sciences'])

model.add_alternate_identifier("https://github.com/leaxp/Deep-Learning-Super-Resolution-Image-Reconstruction-DSIR", 'URL')

#  Add requirements
model.add_requirement('torch', 'detect')
model.add_requirement('torchvision', 'detect')

# Add app.py, which holds the noop function, to the list of associated files to be submitted
model.add_file("app.py")
model.add_file(os.path.join("Deep-Learning-Super-Resolution-Image-Reconstruction-DSIR",
                            "model", "autoencoder_model.pt"))

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('model_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)
