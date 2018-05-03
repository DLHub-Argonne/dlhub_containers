from matminer.featurizers.composition import ElementProperty
import pickle as pkl

f = ElementProperty.from_preset("magpie")
with open('featurizer.pkl', 'rb') as fp:
    pkl.dump(f, fp)