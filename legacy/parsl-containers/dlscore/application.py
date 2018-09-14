import os
import math
import pickle
import textwrap
import numpy as np
import pandas as pd
from keras.models import model_from_json
from util import *

############ CONSTANTS ############
VINA_EXECUTABLE =  "./vina/vina" #If running locally, download the correct system vina executable and change path
NUM_NETWORKS =  10 #Maximum of 10 --> uses the top NUM_NETWORKS models to predict
MODELS_PATH =  "./model"


def run(data):
    lig = data[0]
    rec = data[1]
    receptor = PDB()
    receptor.LoadPDB_from_file(rec)
    receptor.OrigFileName = rec
    f = open(lig,'r')
    lig_array = []
    line = "NULL"
    scores = []
    model_id = 1
    while len(line) != 0:
        line = f.readline()
        if line[:6] != "ENDMDL": lig_array.append(line)
        if line[:6] == "ENDMDL" or len(line) == 0:
            if len(lig_array) != 0 and lig_array != ['']:
                temp_filename = lig + ".MODEL_" + str(model_id) + ".pdbqt"
                temp_f = open(temp_filename, 'w')
                for ln in lig_array: temp_f.write(ln)
                temp_f.close()
                model_name = "MODEL " + str(model_id)
                try:
                    score=calculate_score(lig_array, receptor, VINA_EXECUTABLE, NUM_NETWORKS, temp_filename, rec, "\t", MODELS_PATH)
                except Exception as e:
                    os.remove(temp_filename)
                    raise(e)
                score['dlscore'] = sum(score['dlscore']) / len(score['dlscore'])
                scores.append(score)
                os.remove(temp_filename)
                lig_array = []
                model_id = model_id + 1

    f.close()
    res = pd.DataFrame.to_dict(pd.DataFrame(scores))
    return res


def test_run():
    data_path = "./data"
    inpt = (os.path.join(data_path, "ligand.pdbqt"), os.path.join(data_path, "receptor.pdbqt"))
    output = run(inpt)
    assert type(output) == type({}) #Output should return a dictionary
    assert len(output["dlscore"]) == 12 #Specific to the example data. Remove if example data changes
    print(output)
    return output

if __name__ == '__main__':
    test_run()
