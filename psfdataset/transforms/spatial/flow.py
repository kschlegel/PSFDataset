# -----------------------------------------------------------
# Class to compute the pairwise differences of landmarks between frames.
# This replaces optical flow in image based methods.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np


class Flow:
    """
    Computes the temporal skeleton flow.

    Takes pairwise differences of skeletons between frames. This replaces
    optical flow of image based methods. The flow is computed from the previous
    frame to the current one.

    Methods
    -------
    get_desc()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, keep_location=True):
        """
        Parameters
        ----------
        keep_location: bool, optional (default True)
            If True the transformation returns an array with the last dimension
            twice the size of the input, containing the original locations and
            their flows. The location in the first frame is dropped.
            If False only the flow is returned.
        """
        self._keep_location = keep_location

    def __call__(self, sample):
        flow = np.zeros((sample.shape[0] - 1, ) + sample.shape[1:])
        for i in range(flow.shape[0]):
            flow[i] = sample[i + 1] - sample[i]
        if self._keep_location:
            return np.concatenate((sample[1:], flow), axis=2)
        else:
            return flow

    def get_desc(self):
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(s)Flow": True, "(s)Flow/keep_loc": self._keep_location}
