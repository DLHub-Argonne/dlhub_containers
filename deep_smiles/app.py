import json
import os

# Configurable options
library_path = os.path.join('data', 'character_library.json')
max_length = 226

# Load the library
with open(library_path) as fp:
    library = json.load(fp)


def encode_string(string, length=max_length):
    """Encode a string into a list of integers

    Uses a library of known characters, and gets the index of each character
    in the string into that library.

    Also pads the list with zeros until it reaches a maximum length

    Args:
        string (string): String to encode
        length (int): Desired length of string
    Returns:
        ([int]) Encoded string
    """

    # Get the encoding
    encoded = [library.index(c) for c in string]

    # Check that none of the entries are "not found"
    if any([c == -1 for c in encoded]):
        raise ValueError('String \"{}\" contains character not in library'.format(string))

    # Check if it is not too long
    if len(string) > length:
        raise ValueError('String is too long. {} > {}'.format(len(string), length))

    # Return the padded version
    return encoded + [0] * (length - len(encoded))
