"""
Script to predict the energy and forces of each atom of a C20 molecule

Taken from: https://github.com/atomistic-machine-learning/SchNet/blob/master/scripts/example_md_predictor.py

"""

from schnet.md import SchNetMD
from ase.io import read
from io import StringIO
import tensorflow as tf
import os


# Create the MD predictor
#  LW 16Sept18: I would invest a way to make these models not hard-coded, but the
#    SchNet package used to create these models is already deprecated
energy_path = os.path.join('SchNet', 'models', 'c20', 'energy_model')
force_path = os.path.join('SchNet', 'models', 'c20', 'force_model')


def run(molecule, relax=False):
    """Evaluate the force and energy of a molecule

    Args:
        molecule (string): Molecule in XYZ format
        relax (bool): Whether to determine the energy of the relaxed structure
        """
    with StringIO(molecule) as fp:
        at = read(fp, format='xyz')

    # Before loading the model in a second time, we have to reset TensorFlow
    tf.reset_default_graph()  # Allow for multiple runs

    # Predictor has to be re-created with each invocations because it needs to
    #  be given the charges, and those are not known until the model is invoked
    mdpred = SchNetMD(energy_path, force_path, nuclear_charges=at.numbers)

    # If desired, relax the structure
    if relax:
        energy, forces = mdpred.get_energy_and_forces(at.positions)
        eq_pos = mdpred.relax(at.positions, eps=1e-4, rate=5e-4)
        at.set_positions(eq_pos)

    # Compute the energy of the structure
    energy, forces = mdpred.get_energy_and_forces(at.positions)

    return {'energy': energy, 'forces': forces}


def test_run():
    # Read in an example file
    example_file = os.path.join('SchNet', 'models', 'c20', 'C20.xyz')
    with open(example_file, 'r') as fp:
        example_input = fp.read()

    # Run it
    run(example_input)
    run(example_input, relax=True)

if __name__ == "__main__":
    test_run()
