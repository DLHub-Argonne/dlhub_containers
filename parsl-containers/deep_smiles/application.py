import numpy as np
import pandas as pd
from kipoi.model import KerasModel
from keras.preprocessing import sequence

char_lib_path = "./model/char_lib.npy"
model_path = "./model/CYP1A2_conv1.hdf5"
maxlen=226 # May change, taken from max length feature of training/testing data

def run(data):
    if not isinstance(data, str):
        raise("Expected input is a string to a file. Instead got type {}".format(type(data)))

    ext = data.split(".")[-1]
    if ext == "smi": #Input is a path to a smiles file
        pre_np = pd.read_csv(data)
        data = np.array([val.values[0] for _, val in pre_np.iterrows()])
    elif ext == "npy": # Input is a numpy array
        data = np.load(data)
    else:
        raise("Unexpected file extension. Only .smi and .npy are supported")

    char_lib = np.load(char_lib_path)

    #SMILES sequence to an array
    data_arr = []
    for SMILESsequence in data:
        sequence_arr = []
        for letter in SMILESsequence:
            r = np.where(char_lib == letter)
            sequence_arr.append(np.where(char_lib == letter)[0][0])
        data_arr.append(sequence_arr)
    data_arr = sequence.pad_sequences(data_arr, maxlen = maxlen)

    model = KerasModel(model_path)
    res = model.predict_on_batch(data_arr)

    return res


def test_run():
    data_path = "./data/test_smiles.npy"

    output = run(data_path)

    print(output)
    return output


if __name__ == '__main__':
    test_run()
