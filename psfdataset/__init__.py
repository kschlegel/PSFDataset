# -----------------------------------------------------------
# Package init file for PSFDatasets.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
"""
This package defines tools to create and handle datasets using the path
signature feature methodology developped in
https://arxiv.org/abs/1707.03993

Classes
-------
PSFDataset
    Class to create and handle path-signature feature datasets
PSFZippedDataset
    Class to perform standard zipping of PSFDatasets, while handling their
    additional structure.

Subpackages
-----------
transforms
    Classes to compute the various transformations which may be used as part of
    the path-signature feature methodology.
"""
import numpy as np

from .psfdataset import PSFDataset
from .psfzippeddataset import PSFZippedDataset
from . import transforms
