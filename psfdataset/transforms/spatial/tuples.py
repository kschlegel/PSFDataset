# -----------------------------------------------------------
# Class to form all (unordered) tuples of given size of landmarks.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np
from itertools import combinations

from ...types import DescriptionDict


class Tuples:
    """
    From all (unordered) n-tuples of landmarks.

    Takes an array of the form [frame][landmark][coords] and for every frame
    forms every possible (unordered) n-tuples of landmarks. Returns an array
    [frame][tuple][landmark][coords] of n-tuples of landmarks.

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, tuple_size: int) -> None:
        """
        Parameters
        ----------
        tuple_size : int
            size of tuples to be formed
        """
        self._tuple_size = tuple_size

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        tuples = []
        for frame in range(sample.shape[0]):
            tuples_frame = []
            for tup in combinations(range(sample.shape[1]), self._tuple_size):
                tuples_frame.append(sample[frame][list(tup)])
            tuples.append(tuples_frame)
        return np.array(tuples)

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(SP)Tup": self._tuple_size}
