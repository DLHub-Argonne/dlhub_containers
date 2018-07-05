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

    if data.split(".")[-1] != "npy":
        raise("Unexpected file extension. Only .npy is supported")
    char_lib = np.load(char_lib_path)
    data = np.load(data)

    #SMILES sequence to an array
    data_arr = []
    for SMILESsequence in data: # Might syntactically change based on input data
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
