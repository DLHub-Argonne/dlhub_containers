import os
import json
import keras
import numpy as np
from cellCnn.model import build_model
from sklearn.externals import joblib

######### CONSTANTS #########
results_path = "./model/results_ungated.json"
scaler_path = "./model/scaler_ungated.save"
model_path = "./model/"
best_3_models = [3, 7, 6] # taken from the best models and stored in models folder

def run(data):
    with open(results_path, 'r') as f:
        results = json.load(f)
    scaler = joblib.load(scaler_path)

    new_samples = [scaler.transform(x) for x in data]
    ncell_per_sample = np.min([x.shape[0] for x in new_samples])
    nmark = new_samples[0].shape[1]
    nfilter = results['config']['nfilter'][3] # 3 == best_model
    ncell_pooled = max(1, int(results['config']['maxpool_percentage'][3]/100. * ncell_per_sample))
    regression = False
    n_classes = results['n_classes']

    new_samples = [x[:ncell_per_sample].reshape(1, ncell_per_sample, nmark) for x in new_samples]
    data_test = np.vstack(new_samples)

    y_pred = np.zeros((3, len(new_samples), n_classes))
    for j, i in enumerate(best_3_models):
        print ('Predictions based on multi-cell inputs containing %d cells.' % ncell_per_sample)
        nfilter = results['config']['nfilter'][i]
        maxpool_percentage = results["config"]['maxpool_percentage'][i]
        ncell_pooled = max(1, int(maxpool_percentage/100. * ncell_per_sample))
        temp_model = build_model(ncell_per_sample, nmark,
                        nfilter=nfilter, coeff_l1=0, coeff_l2=0, coeff_activity=0,
                        k=ncell_pooled, dropout=False, dropout_p=0,
                        regression=regression, n_classes=n_classes, lr=0.01)

        weight_path = os.path.join(model_path, "nnet_run_{}.hdf5".format(i))
        temp_model.load_weights(weight_path)

        y_pred[j] = temp_model.predict(data_test)

    res = np.mean(y_pred, axis=0)

    return res


def test_run():
    data_path = "./data/test_ungated.npy"
    labels_path = "./data/test_labels_ungated.npy"
    data = np.load(data_path, encoding="bytes")
    labels = np.load(labels_path)

    output = run(data)

    assert len(output) == len(data)

    print("Prediction: ", output)
    print("True Values: ", labels)
    return output


if __name__ == "__main__":
    test_run()
