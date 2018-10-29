from dlhub_sdk.utils.schemas import validate_against_dlhub_schema
from dlhub_sdk.models.servables.keras import KerasModel
import json


# Describe the keras model
model = KerasModel.create_model('model.h5', list(map(str, range(10))))

#    Describe the model
model.set_title("MNIST Digit Classifier")
model.set_name("mnist")
model.add_alternate_identifier("https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py",
                               "URL")
model.set_domains(["digit recognition"])

#    Describe the outputs in more detail
model.output['description'] = 'Probabilities of being 0-9'
model.input['description'] = 'Image of a digit'

# Print out the result
# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('model_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)
