# -----------------------------------------------------------
# Classes to normalize keypoints into [-1,1]. Either for pure spatial
# coordinate arrays or for arrays containing confidence scores. In the latter
# case confidence scores can be left untouched or normalized separately.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
from typing import Optional, Tuple, Dict

import numpy as np

from ...types import DescriptionDict


class Normalize:
    """
    Normalize all coordinates to [-1,1].

    Takes an array of the form [frame,landmark,coords] where coords only
    contains spatial coordinates and normalizes all values into [-1,1].

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self,
                 data_max: Optional[int] = None,
                 data_min: int = 0) -> None:
        """
        Parameters
        ----------
        data_max: int, optional (default None)
            If None the data range is automatically inferred at each call
            If numeric value the data is assumed to be always less or equal
            than this value in every dimension.
        data_min: int, optional (default 0)
            Ignored and automatically inferred from the data if data_max is
            None. Otherwise the data is assumed to be bigger or equal than this
            value in every dimension.
        """
        self._factor: Optional[float]
        self._shift: Optional[float]
        if data_max is not None:
            self._factor, self._shift = self._compute_params(
                data_max, data_min)
        else:
            self._factor = None
            self._shift = None

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if self._factor is None or self._shift is None:
            data_max = np.amax(sample)
            data_min = np.amin(sample)
            factor, shift = self._compute_params(data_max, data_min)
        else:
            factor = self._factor
            shift = self._shift
        transformed = sample * factor
        transformed -= shift
        return transformed

    def _compute_params(self, data_max: int,
                        data_min: int) -> Tuple[float, float]:
        data_range = data_max - data_min
        if data_range == 0:
            return 1, 0
        factor = 2 / data_range
        shift = 2 * data_min / data_range + 1
        return factor, shift

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        desc: Dict = {"(s)Normalize": "all"}
        if self._factor is not None:
            desc["(s)Normalize/factor"] = self._factor
            desc["(s)Normalize/shift"] = self._shift
        return desc


class NormalizeWithoutConfidence(Normalize):
    """
    Takes spatial+confidence coords, normalize spatial coordinates to [-1,1].

    Takes an array of the form [frame,landmark,coords] where coords contains
    spatial coordinates and a confidence value as last dimension. Normalizes
    the spatial coordinates into [-1,1] and leaves the confidence values
    untouched in [0,1].

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        transformed = super().__call__(sample[:, :, :-1])
        transformed = np.concatenate(
            (transformed, sample[:, :, -1].reshape(sample.shape[:-1] + (1, ))),
            axis=2)
        return transformed

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        desc = super().get_description()
        desc.update({"(s)Normalize": "coords"})
        return desc


class NormalizeWithConfidence(NormalizeWithoutConfidence):
    """
    Takes spatial+confidence coords, normalize all coordinates to [-1,1].

    Takes an array of the form [frame,landmark,coords] where coords contains
    spatial coordinates and a confidence value as last dimension. Normalizes
    the spatial coordinates into [-1,1] and separatelty also normalizes the
    confidence values into [-1,1].

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        transformed = super().__call__(sample)
        transformed[:, :, -1] *= 2
        transformed[:, :, -1] -= 1
        return transformed

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        desc = super().get_description()
        desc.update({"(s)Normalize": "coords+confidence"})
        return desc
