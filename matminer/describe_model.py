from dlhub_toolbox.models.servables.python import PythonClassMethodModel, PythonStaticMethodModel
from dlhub_toolbox.utils.schemas import validate_against_dlhub_schema
from dlhub_toolbox.models.servables.sklearn import ScikitLearnModel
from pymatgen import Composition
import pickle as pkl
import json


# Generate a servable for the first step: generating a list of Python objects from string
convert = PythonStaticMethodModel.from_function_pointer(Composition, autobatch=True)
convert.set_title("Convert List of Strings to Pymatgen Composition Objects")
convert.set_name('string_to_pmg_composition')
convert.set_inputs("list", "List of strings", item_type="string")
convert.set_outputs("list", "List of pymatgen composition objects",
                    item_type={'type': 'python object',
                                    'python_type': 'pymatgen.core.Composition'})
convert.add_requirement('pymatgen', 'latest')


# Generate a servable for the second step: converting features
with open('featurizer.pkl', 'rb') as fp:
    featurizer = pkl.load(fp)
feat_info = PythonClassMethodModel('featurizer.pkl', 'featurize_many', {'ignore_errors': True})

#   Add reference information
feat_info.set_title('Composition featurizer of Ward et al. 2016')
feat_info.set_name("ward_npj_2016_featurizer")
feat_info.set_authors(['Ward, Logan'], ['University of Chicago'])

#   Add citation information
feat_info.add_related_identifier('10.1038/npjcompumats.2016.28', 'DOI', 'IsDescribedBy')

#   Describe the software requirements
feat_info.add_requirement('matminer', 'detect')

#   Describe the inputs and outputs
feat_info.set_inputs('list', 'List of pymtagen Composition objects',
                     item_type={'type': 'python object', 'python_type': 'pymatgen.core.Composition'})
feat_info.set_outputs('ndarray', 'List of features', shape=[None, len(featurizer.feature_labels())])

# Make the model information
model_info = ScikitLearnModel('model.pkl', n_input_columns=len(featurizer.feature_labels()))

#    Describe the model
model_info.set_title("Formation enthalpy predictor")
model_info.set_name("delta-e_icsd-subset_model")
model_info.set_domains(["materials science"])

# Print out the result for each component
def dump_metadata(model, path):
    metadata = model.to_dict()
    print(json.dumps(metadata, indent=2))
    validate_against_dlhub_schema(metadata, 'servable')
    with open(path, 'w') as fp:
        json.dump(metadata, fp, indent=2)

dump_metadata(convert, 'convert_metadata.json')
dump_metadata(feat_info, 'featurizer_metadata.json')
dump_metadata(model_info, 'model_metadata.json')
