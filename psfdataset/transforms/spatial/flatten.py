# -----------------------------------------------------------
# Class to flatten the feature set array into a two dimensional array,
# preserving only the first original dimension.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------


class Flatten:
    """
    Flattens the feature set array into a 2d array.

    Preserves the first dimension of the input array and flattens all remaining
    dimensions into one dimension.

    Methods
    -------
    get_desc()
        Return a dictionary describing the properties of the transformation.
    """
    def __call__(self, sample):
        transformed = sample.reshape((sample.shape[0], -1))
        return transformed

    def get_description(self):
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(s)Flatten": True}

    def explain(self, input_structure):
        output_structure = [
            input_structure[0],
            [
                lm + " " + str(c) for lm in input_structure[1]
                for c in input_structure[2]
            ]
        ]
        return output_structure
