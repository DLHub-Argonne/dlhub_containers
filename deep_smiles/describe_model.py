from dlhub_toolbox.models.servables.python import PythonClassMethodModel, PythonStaticMethodModel
from dlhub_toolbox.utils.schemas import validate_against_dlhub_schema
from dlhub_toolbox.utils.types import compose_argument_block
from dlhub_toolbox.models.servables.keras import KerasModel
import json
import os

# Describe the deep learning model
model = KerasModel(os.path.join('Deep-SMILES', 'CYP1A2_conv1_sequential_best.hdf5'), ['Yes', 'No'])

#  Describe the inputs and outputs
model.input['description'] = 'Encoding of the characters at each point in a string, padded by zeros'
model.output['description'] = 'Binary classification of molecule'

#  Add provenance information
model.set_authors(["Zhu, Mengyuan"], ["Georgia State University"])
model.set_title("Classification Model for AMDET Properties")
model.set_name("deep-smiles_model")
model.set_abstract("A deep learning model that predicts AMDET properties given a SMILES string of a molecule.")
model.add_alternate_identifier("https://github.com/MengyuanZhu/Deep-SMILES", "URL")

#  Add requirements
model.add_requirement('tensorflow', 'detect')
model.add_requirement('keras', 'detect')

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('model_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)


# Describe the encoding step
#  The first step is to turn a string into a list of integers
string_length = model.input['shape'][-1]
model = PythonStaticMethodModel('app', 'encode_string', function_kwargs={'length': string_length},
                                autobatch=True)

#  Describe the inputs and outputs
model.set_inputs('list', 'List of SMILES strings', item_type='string')
model.set_outputs('list', 'List of encoded strings.',
                  item_type=compose_argument_block('list',
                                                   'Encoded string. List of integers where each '
                                                   'value is the index of the character in the '
                                                   'library, or 0 if it is padded',
                                                   item_type='integer'))
#  Add provenance information
model.set_authors(["Zhu, Mengyuan"], ["Georgia State University"])
model.set_title("String Encoder for Classification Model for AMDET Properties")
model.set_name("deep-smiles_enocoder")
model.set_abstract("String encoding step for Deep-SMILES model")
model.add_alternate_identifier("https://github.com/MengyuanZhu/Deep-SMILES", "URL")

#  Add the library and file to be run
model.add_files(['app.py', os.path.join('data', 'character_library.json')])

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('encoder_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)
