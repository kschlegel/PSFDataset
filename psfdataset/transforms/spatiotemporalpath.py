# -----------------------------------------------------------
# Class to transform a spatial into a spatiotemporal path.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from ..types import DescriptionDict


class SpatioTemporalPath:
    """
    A class to transform a spatial into a spatiotemporal path.

    Data usually comes in the format [frame_id,element,coords] or if we formed
    tuples [frame_id,tuple,landmark,coords]. To build a Spatiotemporal dataset
    we need the time evolution of each element, i.e. [element,frame_id,coords].
    This transformations will rearrange the data in this way.

    If the elements are signatures of tuples the last dimension tends to be
    too large and we need to disintegrate into single signature components
    or the dimension will grow too fast in the Spatiotemporal dataset, i.e.
    we want [signature_component, frame_id, 1]

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, disintegrate: bool = True) -> None:
        """
        Parameters
        ----------
        disintegrate : bool
            Whether or not to disintegrate the last (coordinate) dimension into
            its seperate components to avoid explosion of dimensionality later.
        """
        self._disintegrate = disintegrate

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if self._disintegrate:
            sample = sample.reshape(sample.shape[0], -1, 1)
        elif len(sample.shape) > 3:
            # if the data path at this point is tuples of landmarks then it is
            # of the form [frame_id,tuple,landmark,coords], flatten the last 2
            # dimensions into 1
            sample = sample.reshape(sample.shape[0:2] + (-1, ))
        return np.transpose(sample, (1, 0, 2))

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"SpatioTemporalPath/disintegrate": self._disintegrate}
