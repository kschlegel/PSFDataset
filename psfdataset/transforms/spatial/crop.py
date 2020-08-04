# -----------------------------------------------------------
# Class to crop a sequence of keypoints to their common spatial bounding box.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from ...types import DescriptionDict


class Crop:
    """
    Crop the keypoints to their (spatial) bounding box.

    Crop takes an input of the form [frame,landmark,coords] and translates
    the spatial coordinates so that the top, leftmost landmark is at (0,0).
    Thus np.amax will return the exact bounding box.

    If the the landmarks have a confidence score as last dimension a confidence
    of 0 usually indicates missing data (i.e which could not be detected). The
    confidence value can be used to ignore those missing data point (which
    usually are equal to 0).

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, ignore_missing: bool = False) -> None:
        self._ignore_missing = ignore_missing

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if self._ignore_missing:
            mins = np.full(sample.shape[2] - 1, np.inf)
            for frame in range(sample.shape[0]):
                for landmark in range(sample.shape[1]):
                    if sample[frame][landmark][-1] != 0:
                        mins = np.minimum(mins, sample[frame][landmark][0:-1])
            transformed = np.zeros(sample.shape, sample.dtype)
            for frame in range(sample.shape[0]):
                for landmark in range(sample.shape[1]):
                    if sample[frame][landmark][-1] != 0:
                        transformed[frame][landmark][0:-1] = \
                            sample[frame][landmark][0:-1] - mins
                        transformed[frame][landmark][-1] = \
                            sample[frame][landmark][-1]
        else:
            mins = np.amin(sample, axis=(0, 1))
            transformed = sample - mins
        return transformed

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        desc: DescriptionDict = {"(s)Crop": True}
        # this is only attached when True because for datasets without
        # confidence score this setting should always be False
        if self._ignore_missing:
            desc["(s)crop/ignore_missing"] = True
        return desc
