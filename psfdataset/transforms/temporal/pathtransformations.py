# -----------------------------------------------------------
# Classes to compute time incorporated and invisibility reset transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from ...types import DescriptionDict


class TimeIncorporatedTransformation:
    """
    Compute the time incorporated transformation of the path.

    Takes an array of the form [element][frame][coords].
    For each element transforms the temporal path [frame][coords],
    adding an extra dimension for time.
    Example:
    The path 2,8,4 turns into
        (2,0),(8,1),(4,2)
    """
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        time_dimension = np.array([[[j / (len(sample[i]) - 1)]
                                    for j in range(len(sample[i]))]
                                   for i in range(len(sample))],
                                  dtype=np.float64)
        return np.concatenate((sample, time_dimension), axis=2)

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(t)TIT": True}


class InvisibilityResetTransformation:
    """
    Computes the invisibility reset transformation of the path.

    Takes an array of the form [element][frame][coords].
    For each element transforms the temporal path [frame][coords],
    adding a visibility dimension and two extra time steps. The visibility
    coordinate is set to 1 for any original step of the path, and 0 for the
    two new steps. The first of the two new steps copys the last step of the
    original path, the second one is equal to zero.
    Example:
    The Path 1,2,3 turns into
        (1,1),(2,1),(3,1),(3,0),(0,0)
    """
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        extended = np.concatenate(
            (sample,
             np.ones(
                 (sample.shape[0], sample.shape[1], 1), dtype=sample.dtype)),
            axis=2)
        extended = np.concatenate(
            (extended, extended[:, -1:, :],
             np.zeros((extended.shape[0], 1, extended.shape[2]),
                      dtype=sample.dtype)),
            axis=1)
        extended[:, -2, -1] = 0
        return extended

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(t)IRT": True}
