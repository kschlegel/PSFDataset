# -----------------------------------------------------------
# Class to allow chaining of transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
from typing import Sequence

import numpy as np

from ..types import KeypointTransformation, DescriptionDict


class Compose:
    """
    A class to allow chaining of transformations.

    Takes a list of transformations to be applied to the data consecutively.
    This is currently intendened to only be used once to chain several
    transformations, do not recurse compositions (recursive applications should
    still apply transformations correctly, but will break the description dict.
    This may be fixed at a later time).

    Methods
    -------
    get_description()
        Return a dictionary describing the properties of the transformations.
    """
    def __init__(self, transforms: Sequence[KeypointTransformation]) -> None:
        """
        Parameters
        ----------
        transforms : collection of transform objects
            Collection of transformations to be applied consecutively
        """
        self._transforms = transforms

    def __call__(self, data: np.ndarray) -> np.ndarray:
        for transform in self._transforms:
            data = transform(data)
        return data

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of the transformations.

        Collates all the transformations description dicts and adds an entry
        for their order of application.

        Returns
        -------
        dict
            Description of the transformations
        """
        desc: DescriptionDict = {}
        order = ""
        for transform in self._transforms:
            desc.update(transform.get_description())
            # __class__ property is <class 'a.b.c.d'>
            # cut off clutter at start&end end split by module structure
            cls = str(transform.__class__)[8:-2].split(".")
            if len(cls) > 3:  # from a submodule -> add which one
                order += "(" + cls[-3][0:1] + ")"
            order += cls[-1] + "->"
            # Add order string minus the extra arrow at the end
        desc["compose"] = order[:-2]
        return desc
