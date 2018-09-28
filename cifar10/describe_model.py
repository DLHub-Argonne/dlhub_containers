from dlhub_toolbox.models.servables.python import PythonClassMethodModel
from dlhub_toolbox.utils.schemas import validate_against_dlhub_schema
from dlhub_toolbox.utils.types import compose_argument_block
from dlhub_toolbox.models.servables.keras import KerasModel
import json
import os

cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# Describe the deep learning model
model = KerasModel(os.path.join('models', 'cifar10vgg.h5'), cifar_classes)

#  Describe the inputs and outputs
model.set_inputs('list', 'List of images. Each image must be standardized by the mean'
                         ' and standard deviation of the training set',
                 item_type=compose_argument_block('ndarray', 'Image', shape=[32, 32, 3]))
model.set_outputs('ndarray', 'Probabilities of being in each of the cifar classes',
                  shape=[None, 10])

#  Add provenance information
model.set_authors(["Geifman, Yonatan"], ["Technion"])
model.set_title("Keras Model for Cifar10 based on VGGNet")
model.set_name("cifar10_model")
model.set_abstract("A deep learning model that labels images as 10 different common objects (e.g., cats). "
                   "Trained using the CIFAR10 dataset and based on the VGG16 architecture. Achieves an accuracy of"
                   "~90% on the benchmark provided in Keras.")
model.add_alternate_identifier("https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py", "URL")
model.add_related_identifier("1409.1556", "arXiv", "IsDescribedBy")

#  Add requirements
model.add_requirement('tensorflow', 'detect')
model.add_requirement('keras', 'detect')

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('model_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)


# Describe the standardization step
model = PythonClassMethodModel(os.path.join('models', 'img_normalizer.pkl'), 'standardize')

#  Describe the inputs and outputs
model.set_inputs('list', 'List of images',
                 item_type=compose_argument_block('ndarray', 'Image', shape=[32, 32, 3]))
model.set_outputs('list', 'List of images. Standardized from the training set',
                  item_type=compose_argument_block('ndarray', 'Image', shape=[32, 32, 3]))

#  Add provenance information
model.set_authors(["Geifman, Yonatan"], ["Technion"])
model.set_title("Keras Model for Cifar10 based on VGGNet")
model.set_name("cifar10_standardizer")
model.set_abstract("The image standardization routine associated with the cifar10_model deep learning model.")
model.add_alternate_identifier("https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py", "URL")
model.add_related_identifier("1409.1556", "arXiv", "IsDescribedBy")

#  Add requirements
model.add_requirement('keras', 'detect')

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('standardizer_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)
