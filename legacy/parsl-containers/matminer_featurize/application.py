import json
import codecs
import pandas as pd
import pickle as pkl
from home_run.models import BaseHRModel
from matminer.featurizers.base import BaseFeaturizer


class MatminerFeaturizer(BaseHRModel):
    """Model step that uses matminer to compute the features of an object

    Features are stored in a key 'features'"""

    def __init__(self, pickle_path, pickled_columns=None, input_columns=None):
        """Define a featurizer

        Args:
            pickle_path - str, path to pickle file containing featurizers
            pickled_columns - [str], columns that need to be unpickled
            input_columns - [str], columns that are input to the featurizer
        """
        # Load in the featurizer
        with open(pickle_path, 'rb') as fp:
            self.featurizer = pkl.load(fp)
        if not isinstance(self.featurizer, BaseFeaturizer):
            raise Exception('This file is not a featurizer')

        self.pickled_columns = [] if pickled_columns is None else pickled_columns
        self.input_columns = [] if input_columns is None else input_columns

    def run(self, inputs):
        """Run the featurization

        Reads from the columns specified in `self.input_columns`,
        produces a key 'features' that contains a list of features"""
        # Get the desired column as a pd.Series
        data = pd.DataFrame(inputs)

        # If unpickling needed
        for column in self.pickled_columns:
            data[column] = data[column].map(lambda x: pkl.loads(
                codecs.decode(x.encode(), 'base64')))

        # Run the featurizer
        new_columns = self.featurizer.featurize_dataframe(
            data,
            self.input_columns
        )

        # Add features to input
        for (rid, row), i in zip(new_columns[self.featurizer.feature_labels()].iterrows(),
                                 inputs):
            i['features'] = row.values.tolist()

        return inputs


def predict(data):
    with open('options.json') as fp:
        options = json.load(fp)

    ml = MatminerFeaturizer(**options)

    return ml.run(data)
    # return json.dumps(ml.run(data))

def test_run():
    data = [{'composition': 'Al2O3', 'composition_object': 'gANjcHltYXRnZW4uY29yZS5jb21wb3NpdGlvbgpDb21wb3NpdGlvbgpxACmBcQF9cQIoWA4AAABh\nbGxvd19uZWdhdGl2ZXEDiVgHAAAAX25hdG9tc3EER0AUAAAAAAAAWAUAAABfZGF0YXEFfXEGKGNw\neW1hdGdlbi5jb3JlLnBlcmlvZGljX3RhYmxlCkVsZW1lbnQKcQdYAgAAAEFscQiFcQlScQpHQAAA\nAAAAAABoB1gBAAAAT3ELhXEMUnENR0AIAAAAAAAAdXViLg==\n'},
        {'composition': 'NaCl', 'composition_object': 'gANjcHltYXRnZW4uY29yZS5jb21wb3NpdGlvbgpDb21wb3NpdGlvbgpxACmBcQF9cQIoWA4AAABh\nbGxvd19uZWdhdGl2ZXEDiVgHAAAAX25hdG9tc3EER0AAAAAAAAAAWAUAAABfZGF0YXEFfXEGKGNw\neW1hdGdlbi5jb3JlLnBlcmlvZGljX3RhYmxlCkVsZW1lbnQKcQdYAgAAAE5hcQiFcQlScQpHP/AA\nAAAAAABoB1gCAAAAQ2xxC4VxDFJxDUc/8AAAAAAAAHV1Yi4=\n'}]
    with open('options.json') as fp:
        options = json.load(fp)

    ml = MatminerFeaturizer(**options)

    return ml.run(data)
