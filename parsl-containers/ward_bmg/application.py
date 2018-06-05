import gzip
import pickle as pkl
import matplotlib
matplotlib.use('TkAgg')
from ternary.helpers import normalize, simplex_iterator
from pymatgen import Composition

model_path = "./model/model.pkl.gz"
featurizer_path = "./model/featurizer.pkl.gz"
label_path = "./model/label_enc.pkl"

def run(data, version1=True):
    with gzip.open(model_path, 'rb') as fp:
        model = pkl.load(fp)
    with gzip.open(featurizer_path, 'rb') as fp:
        featurizer = pkl.load(fp)
    with open(label_path, 'rb') as fp:
        label_enc = pkl.load(fp)


    ############## Version 1 ################
    if version1:
        scale=10
        boundary=True
        prediction = dict()
        for i, j, k in simplex_iterator(scale=scale, boundary=boundary):
            prediction[(i, j)] = get_coords(data, normalize([i, j, k]), featurizer, model)
        return prediction


    ############## Version 2 ################
    else:
        X = featurizer.featurize(Composition(data))
        return model.predict_proba([X])

def test_run(version1=True):

    ############## Version 1 ################
    if version1:
        X = ('Zr', 'Cr', 'Fe')

    ############## Version 2 ################
    else:
        X = "Cr0.0625 Fe0.9375"
    output = run(X, version1)

    print(output)
    return output


def get_coords(elems, x, featurizer, model):
    comp = Composition(dict(zip(elems, x)))
    try:
        X = featurizer.featurize(comp)
        return model.predict_proba([X])[0][0]
    except:
        return 0

if __name__ == '__main__':
    test_run()
