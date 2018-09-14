from kipoi.model import SklearnModel
import numpy as np

model_path = "./model/ward_formation_energy.pkl"

def run(data):
    model = SklearnModel(model_path)
    return model.predict_on_batch(data)

def test_run():
    X = np.load("./data/test_data.npy")
    output = run(X)

    assert len(output) == len(X)

    print(output)
    return output
