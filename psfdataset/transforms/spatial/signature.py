# -----------------------------------------------------------
# Class to take signatures of tuples of landmarks before converting into
# spatiotemporal paths.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np
from esig import tosig

from ...types import DescriptionDict


class Signature:
    """
    Takes signatures of tuples of landmarks.

    Takes an array of the form [frame][tup][landmark][coords] where tup denotes
    a tuples of landmarks. The tuple is considered a path in space, i.e. a
    3-tuple is a path in #coords dimensional space with 3 datapoints.
    The signature of this path is computed and returned as an array of the form
    [frame][tup][signature_component].

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self,
                 signature_level: int,
                 drop_zeroth_term: bool = True) -> None:
        """
        Parameters
        ----------
        signature_level : int
            level of the signature to be computed
        drop_zeroth_term : bool, optional (default is True)
            whether or not to drop the zeroth term of the signature (which is
            always equal to 1)
        """
        self._signature_level = signature_level
        self._drop_zeroth_term = drop_zeroth_term

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        signatures = []
        for frame in range(sample.shape[0]):
            signatures_frame = []
            for tup in range(sample.shape[1]):
                signature = tosig.stream2sig(
                    sample[frame][tup].astype(np.float64),
                    self._signature_level)
                signatures_frame.append(signature[self._drop_zeroth_term:])
            signatures.append(signatures_frame)
        return np.array(signatures)

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {
            "(s)Sig/lvl": self._signature_level,
            "(s)Sig/drop_zeroth": self._drop_zeroth_term
        }
