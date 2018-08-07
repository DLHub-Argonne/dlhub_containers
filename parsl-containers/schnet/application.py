import os
import ase.io
import numpy as np
import tensorflow as tf
from utils.util import load_model, get_atom_indices

###### CONSTANTS ######
base_path = "./models/"
energy_model_path = os.path.join(base_path, "energy_model")
force_model_path = os.path.join(base_path, "force_model")
batch_size = 1
allow_local_file = False # Set True if running locally for simpler use


def run(data):
    temp_name = "temp_data/temp_file.xyz"
    if not allow_local_file: # Not a local path. Data must be read in
        if os.path.isfile(temp_name):
            os.remove(temp_name)
        with open(temp_name, 'w') as temp_f:
            temp_f.write(data)
        data = temp_name
    tf.reset_default_graph() # Allow for multiple runs
    at = ase.io.read(data)
    if os.path.isfile(temp_name): # Keep temporary directory clean
        os.remove(temp_name)
    energy_model = load_model(energy_model_path)
    force_model = load_model(force_model_path)

    nuclear_charges = at.numbers
    n_atoms = len(nuclear_charges)
    charges = tf.tile(tf.constant(nuclear_charges.ravel(), dtype=tf.int64), (batch_size,))
    positions = tf.placeholder(tf.float32, shape=(batch_size * n_atoms, 3))
    seg_m, idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j = get_atom_indices(n_atoms, batch_size)

    g = tf.get_default_graph()
    with g.gradient_override_map({"Tile": "TileDense"}):
        energy = energy_model(charges, positions,
                                        offset, idx_ik, idx_jk, idx_j,
                                        seg_m, seg_i, seg_j, ratio_j)
        force_energy = force_model(charges, positions,
                                      offset, idx_ik, idx_jk, idx_j,
                                      seg_m, seg_i, seg_j, ratio_j)
        forces = -tf.reshape(tf.convert_to_tensor(tf.gradients(tf.reduce_sum(force_energy),
                                                                        positions)[0]),
                                      (batch_size, n_atoms, 3))
    ckpt = tf.train.latest_checkpoint(os.path.join(energy_model_path, 'validation'))
    session = tf.Session()
    energy_model.restore(session, ckpt)
    ckpt = tf.train.latest_checkpoint(os.path.join(force_model_path, 'validation'))
    force_model.restore(session, ckpt)

    orig_positions = positions
    positions = at.positions.reshape((-1, 3)).astype(np.float32)
    feed_dict = {
        orig_positions: positions
    }
    E, F = session.run([energy, forces], feed_dict=feed_dict)
    res = {"energy":E.tolist(), "force":F.tolist()} # Allow for json dumps for dlhub service

    return res


def test_run():
    test_data_path = os.path.join("./data", "C20.xyz")
    if not allow_local_file:
        test_data = open(test_data_path, 'r').read()
        output = run(test_data)
    else:
        output  = run(test_data_path)

    energy, force = output["energy"], output["force"]

    print("ENERGY: ", energy)
    print("FORCE: ", force)

    return output


if __name__ == '__main__':
    test_run()
