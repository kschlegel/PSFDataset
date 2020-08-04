# -----------------------------------------------------------
# Classes to (optionally) split a spatiotemporal path into dyadic
# intervals and take signatures of each segment.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
from typing import List

import numpy as np
from esig import tosig

from ...types import DescriptionDict


class DyadicPathSignatures:
    """
    Take signatures of (dyadic intervals of) a spatiotemporal path.

    Takes a spatiotemporal path, i.e. an array of the form
    [element][frame][coords] and optionally splits the frame axis into
    (possibly half overlapping) dyadic intervals. Calculates signatures of each
    piece. Returns [list][list][numpy.ndarray] of the form
    [element][dyadic_piece][signature_terms].

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self,
                 dyadic_levels: int,
                 signature_level: int = 2,
                 overlapping: bool = False) -> None:
        """
        Parameters
        ----------
        dyadic_levels : int
            number of dyadic splits to do. Level 0 corresponds to just the
            original path, 1 is the path and its halfs, etc.
        signature_level: int, optional (default is 2)
            level of signatures to be computed
        overlapping: bool, optional (default is False)
            Whether to take the dyadic intervals half overlapping each other
        """
        self._dyadic_levels = dyadic_levels
        self._signature_level = signature_level
        self._overlapping = overlapping

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        dyadic_pieces: List[List[np.ndarray]] = [
            [] for i in range(sample.shape[0])
        ]
        for dyadic_level in range(self._dyadic_levels + 1):
            if self._overlapping:
                num_pieces = 2**(dyadic_level + 1) - 1
                if num_pieces > 1:
                    frames_per_piece = sample.shape[1] / (num_pieces - 1)
                else:
                    frames_per_piece = sample.shape[1]
            else:
                num_pieces = 2**dyadic_level
                frames_per_piece = sample.shape[1] / num_pieces

            for i in range(sample.shape[0]):
                for j in range(num_pieces):
                    # if overlapping=True only shift by half the
                    # frames_per_piece at a time
                    start_frame = j * frames_per_piece / 2**self._overlapping
                    end_frame = start_frame + frames_per_piece
                    start_frame = int(start_frame)
                    end_frame = int(end_frame)
                    dyadic_pieces[i] += [
                        tosig.stream2sig(
                            sample[i][start_frame:end_frame].reshape(
                                end_frame - start_frame,
                                -1).astype(np.float64), self._signature_level)
                    ]
        return np.array(dyadic_pieces)

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {
            "(t)DySig/dylvl": self._dyadic_levels,
            "(t)DySig/siglvl": self._signature_level,
            "(t)DySig/overlap": self._overlapping
        }
