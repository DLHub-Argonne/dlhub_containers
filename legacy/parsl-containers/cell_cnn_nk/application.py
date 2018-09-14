import json
import keras
import numpy as np
from cellCnn.model import build_model
from sklearn.externals import joblib

######### CONSTANTS #########
results_path = "./model/results.npy"
scaler_path = "./model/scaler.save"
model_path = "./model/nnet_run_3.hdf5"

def run(data):
    #results = json.load("./examples'results.json")
    results = np.load(results_path).tolist()
    scaler = joblib.load(scaler_path)

    new_samples = [scaler.transform(x) for x in data]
    ncell_per_sample = np.min([x.shape[0] for x in new_samples])
    nmark = new_samples[0].shape[1]
    nfilter = results['config']['nfilter'][3] # 3 == best_model
    ncell_pooled = max(1, int(results['config']['maxpool_percentage'][3]/100. * ncell_per_sample))
    regression = False
    n_classes = results['n_classes']

    my_model = build_model(ncell_per_sample, nmark, nfilter=nfilter,
                                    coeff_l1=0, coeff_l2=0, coeff_activity=0,
                                    k=ncell_pooled, dropout=False, dropout_p=0,
                                    regression=regression, n_classes=n_classes, lr=0.01)
    my_model.load_weights(model_path)

    new_samples = [x[:ncell_per_sample].reshape(1, ncell_per_sample, nmark) for x in new_samples]
    data_test = np.vstack(new_samples)
    res = my_model.predict(data_test)

    return res


def test_run():
    data_path = "./data/test.npy"
    labels_path = "./data/test_labels.npy"
    data = np.load(data_path, encoding="bytes")
    labels = np.load(labels_path)

    output = run(data)

    assert len(output) == len(data)

    print("Prediction: ", output)
    print("True Values: ", labels)
    return output


if __name__ == "__main__":
    test_run()
