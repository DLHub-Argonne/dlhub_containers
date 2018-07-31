import torch
import os
import numpy as np
import json
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from pymatgen.core.structure import Structure
from model.model import *

###### CONSTANTS ######
orig_atom_fea_len = 92
nbr_fea_len = 41
atom_fea_len = 64
n_conv = 4
h_fea_len = 32
n_h = 1
model_path = "model/shear-moduli.pth.tar"
feature_path = "model/atom_init.json"

def run(data):
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=atom_fea_len,
                                n_conv=n_conv,
                                h_fea_len=h_fea_len,
                                n_h=n_h, classification=False)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dir_path = data ############################### Pass in dir_path? Or single cif file?
    dataset = CifData(dir_path)
    val_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_pool)
    normalizer = Normalizer(torch.zeros(3))
    normalizer.load_state_dict(checkpoint['normalizer'])

    test_preds = []
    for i, (inpt, target, batch_cif_ids) in enumerate(val_loader):
        input_var = (Variable(inpt[0], volatile=True),
                             Variable(inpt[1], volatile=True),
                             inpt[2],
                             inpt[3])

        output = model(*input_var)
        test_pred = normalizer.denorm(output.data.cpu())
        test_preds += test_pred.view(-1).tolist()

    return test_preds

def test_run():
    test_data_path = "data"
    output = run(test_data_path)

    test_files = [f for f in os.listdir(test_data_path) if ".cif" in f]
    assert len(output) == len(test_files)

    print(output)
    return output

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class CifData(Dataset):
    def __init__(self, dir_path, radius=8, max_num_nbr=12, dmin=0, step=0.2):
        self.root_dir = dir_path
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.dmin=0
        self.step=0.2
        self.cif_list = [f for f in os.listdir(dir_path) if ".cif" in f]
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def set_atom_fea(self, crystal):
        with open(feature_path) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        embedding = {}
        for key, value in elem_embedding.items():
            embedding[key] = np.array(value, dtype=float)
        atom_fea = np.vstack([embedding[(crystal[i].specie.number)] for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        return atom_fea

    def set_nbr_feas(self, crystal):
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea = np.array([list(map(lambda x: x[1], nbr[:self.max_num_nbr])) for nbr in all_nbrs])
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = np.array([list(map(lambda x: x[2], nbr[:self.max_num_nbr])) for nbr in all_nbrs])
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        return nbr_fea, nbr_fea_idx

    def __len__(self):
        return len(self.cif_list)

    def __getitem__(self, idx):
        crystal = Structure.from_file(os.path.join(self.root_dir, self.cif_list[idx]))
        atom_fea = self.set_atom_fea(crystal)
        nbr_fea, nbr_fea_idx = self.set_nbr_feas(crystal)
        target = torch.Tensor([float(0)]) ## When predicting, target should not matter so set to 0
        cif_id = self.cif_list[idx].split(".cif")[0]
        res = (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
        return res


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids

class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

if __name__ == '__main__':
    test_run()
