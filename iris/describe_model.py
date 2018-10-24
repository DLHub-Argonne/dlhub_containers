from dlhub_toolbox.utils.schemas import validate_against_dlhub_schema
from dlhub_toolbox.models.servables.sklearn import ScikitLearnModel
import json

# Initiate the model
model = ScikitLearnModel.create_model('model.pkl', n_input_columns=4,
                                      classes=['setosa', 'versicolor', 'virginica'])

#    Describe the model
model.set_title("SVM to Predict Iris Species")
model.set_abstract('SVM model trained using the classic Iris dataset')
model.set_name("iris_svm")
model.set_domains(["biology"])

# Print out the result
# Sanity Check: Make sure it fits the schema
metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')
with open('model_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)
