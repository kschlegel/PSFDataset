# -----------------------------------------------------------
# Classes to compute lead-lag and multi-delayed transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from ...types import DescriptionDict


class LeadLagTransformation:
    """
    Compute the lead-lag transformation of a path.

    Expects an array of the form [element][frame][coords]. For each element
    forms tuples of path elements, advancing time one component at a time.

    Example:
    The path 1,2,3 with delay 2 turns into
        (1,1,1),(2,1,1),(2,2,1),(2,2,2),(3,2,2),(3,3,2),(3,3,3)
    """
    def __init__(self, delay: int = 1) -> None:
        """
        Parameters
        ----------
        delay : int, optional (default is 1)
            How many timesteps to delay the path
        """
        self._delay = delay

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        lead_lag = np.zeros((sample.shape[0],
                             (self._delay + 1) * sample.shape[1] - self._delay,
                             self._delay + 1, sample.shape[2]),
                            dtype=sample.dtype)
        for i in range(sample.shape[0]):
            for frame in range(sample.shape[1]):
                # Every element is visible for 2*delay+1 frames. The first
                # element starts in the middle of its cycle, i.e. at -delay.
                # From there each elements appears delay+1 times in each row
                # and is shifted by one each row.
                target_frame = (self._delay + 1) * frame - self._delay
                for j in range(self._delay + 1):
                    for k in range(self._delay + 1):
                        if target_frame + j + k >= 0 and target_frame + j + k < lead_lag.shape[
                                1]:
                            lead_lag[i][target_frame + j +
                                        k][k] = sample[i][frame]
        return lead_lag

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(t)LLT": self._delay}


class MultiDelayedTransformation:
    """
    Compute the multi-delayed transformation of a path.

    Expects an array of the form [element][frame][coords]. This is a variant of
    the lead-lag transformation which instead of advancing time one component
    at a time it advances time by one in each component every step so that each
    element of the multi-delayed path contains the last delay elements of the
    original path. Pads with zeros at the ends.

    Example:
    The path 1,2,3 with delay 1 turns into
        (1,0),(2,1),(3,2),(0,3)
    """
    def __init__(self, delay: int = 1) -> None:
        """
        Parameters
        ----------
        delay : int, optional (default is 1)
            How many timesteps to delay the path
        """
        self._delay = delay

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        delayed_path = []
        delayed_path = np.zeros(
            (sample.shape[0], sample.shape[1] + self._delay, self._delay + 1,
             sample.shape[2]),
            dtype=sample.dtype)
        for i in range(sample.shape[0]):
            for frame in range(sample[i].shape[0]):
                for j in range(self._delay + 1):
                    delayed_path[i][frame + j][j] = sample[i][frame]
        return delayed_path

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(t)MDT": self._delay}
