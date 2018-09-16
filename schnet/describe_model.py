from dlhub_toolbox.models.servables.python import PythonStaticMethodModel
from dlhub_toolbox.utils.schemas import validate_against_dlhub_schema
from dlhub_toolbox.utils.types import compose_argument_block
import json
import os

# Create a model that invokes the "run" function from the
model = PythonStaticMethodModel('app', 'run')

#  Describe the inputs and outputs
model.set_title("SchNet C20 Force and Energy Predictor")
model.set_name('schnet_c20')
model.set_abstract("A model based on the SchNet architecture that predicts the energy and forces of a C20 molecule. Useful for molecular dynmaics simulations.")

model.set_inputs('string', 'Molecule in XYZ format')
model.set_outputs('dict', 'Forces and energies of the molecule',
                  properties={
                      'energy': compose_argument_block('number', 'Energy of the whole system'),
                      'forces': compose_argument_block('ndarray', 'Forces acting on each atom in each direction', shape=[None, 3])
                  })

#  Add provenance information
model.add_alternate_identifier("https://github.com/atomistic-machine-learning/SchNet", 'URL')
model.add_related_identifier("1706.08566", "arXiv", "IsDescribedBy")
model.add_related_identifier("1706.08566", "arXiv", "IsDescribedBy")

#  Add requirements
model.add_requirement('ase', 'detect')
model.add_requirement('tensorflow', '1.10.1')   # TODO: Check if this works
model.add_requirement('numpy', 'detect')
model.add_requirement('git+https://github.com/atomistic-machine-learning/SchNet.git')

# Add app.py, which holds the noop function, to the list of associated files to be submitted
model.add_file("app.py")
model.add_directory(os.path.join('SchNet', 'models'), recursive=True)

# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('model_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)
