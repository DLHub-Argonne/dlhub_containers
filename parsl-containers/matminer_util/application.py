import json
import codecs
import pandas as pd
import pickle as pkl
from home_run.models import BaseHRModel
from matminer.utils import conversions


class MatminerConversion(BaseHRModel):
    def __init__(self, input_name, output_name, function_name,
                 unpickle_input=True, pickle_output=True, **kwargs):
        self.input_name = input_name
        self.output_name = output_name
        self.function_name = function_name
        self.unpickle_input = unpickle_input
        self.pickle_output = pickle_output
        self.function_options = kwargs

    def run(self, inputs):
        """Runs a conversion routine from matminer

        Reads from the key specified in `self.input_name` and writes
        to column `self.output_name`"""
        # Get the desired column as a pd.Series
        data = pd.Series([x[self.input_name] for x in inputs])

        # If unpickling needed
        if self.unpickle_input:
            data.map(pkl.loads)

        # Run the desired converter
        f = getattr(conversions, self.function_name)
        data = f(data, **self.function_options)

        # Add data to the output
        for i, o in zip(inputs, data):
            i[self.output_name] = codecs.encode(pkl.dumps(o), 'base64').decode() if self.pickle_output else o

        return inputs


def predict(data):
    # Load in options form disk
    with open('options.json') as fp:
        options = json.load(fp)

    # Instantiate a HRModel for sklearn type
    ml = MatminerConversion(**options)


    return ml.run(data)
    # return json.dumps(ml.run(data))

def test_run():
    data = [
    	{'composition': 'Al2O3'},
	{'composition': 'NaCl'},
    ]
    with open('options.json') as fp:
        options = json.load(fp)

    # Instantiate a HRModel for sklearn type
    ml = MatminerConversion(**options)
    return ml.run(data)
