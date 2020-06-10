# -----------------------------------------------------------
# Package init file for PSFDataset spatial transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
"""
This subpackage defines the various spatial path transformations needed to
create datasets using the path signature feature methodology

Classes
-------
Normalize
    Normalize paths with purely spatial coordinates.
NormalizeWithoutConfidence
    Normalize paths with spatial coordinates and a confidence coordinate.
    Leaves the confidence coordinate untouched.
NormalizeWithConfidence
    Normalize paths with spatial coordinates and a confidence coordinate.
    Normalizes both spatial and confidence coordinates.
Crop
    Crop a sequence of landmarks to their commong spatial bounding box.
RandomSubset
    Extracts a random subset of n frames from a sequence of frames.
FirstN
    Extracts the first n frames from a sequence of frames.
SubSample
    Subsamples every n-th frame from a sequence of frames.
Tuples
    For every frame forms every (unordered) n-tuple of landmarks.
Signature
    Computes signatures of n-tuples of landmarks.
"""

from .normalize import Normalize, NormalizeWithConfidence
from .normalize import NormalizeWithoutConfidence
from .crop import Crop
from .subset import RandomSubset, FirstN, SubSample
from .tuples import Tuples
from .signature import Signature
