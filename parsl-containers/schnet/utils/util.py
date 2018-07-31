import os
import numpy as np
import tensorflow as tf
from .helpers import SchNet

def load_model(model_path):
    args = np.load(os.path.join(model_path, 'args.npy')).item()
    model = SchNet(args.interactions, args.basis, args.filters, args.cutoff,
                   intensive=args.intensive, filter_pool_mode=args.filter_pool_mode)
    return model


def get_atom_indices(n_atoms, batch_size):
    n_distances = n_atoms ** 2 - n_atoms
    seg_m = np.repeat(range(batch_size), n_atoms).astype(np.int32)
    seg_i = np.repeat(np.arange(n_atoms * batch_size), n_atoms - 1).astype(np.int32)
    idx_ik = seg_i
    idx_j = []
    for b in range(batch_size):
        for i in range(n_atoms):
            for j in range(n_atoms):
                if j != i:
                    idx_j.append(j + b * n_atoms)

    idx_j = np.hstack(idx_j).ravel().astype(np.int32)
    offset = np.zeros((n_distances * batch_size, 3), dtype=np.float32)
    ratio_j = np.ones((n_distances * batch_size,), dtype=np.float32)
    seg_j = np.arange(n_distances * batch_size, dtype=np.int32)

    seg_m, idx_ik, seg_i, idx_j, seg_j, offset, ratio_j = \
        tf.constant(seg_m), tf.constant(idx_ik), tf.constant(seg_i), tf.constant(idx_j), \
        tf.constant(seg_j), tf.constant(offset), tf.constant(ratio_j)
    idx_jk = idx_j
    return seg_m, idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j
