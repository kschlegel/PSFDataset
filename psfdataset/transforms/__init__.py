# -----------------------------------------------------------
# Package init file for PSFDataset transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
"""
This subpackage defines the various path transformations needed to create
datasets using the path signature feature methodology

Classes
-------
Compose
    Class to allow chaining of transformations.
SpatioTemporalPath
    Transforms a spatial path into a spatiotemporal path.

Subpackages
-----------
spatial
    Spatial transformations to be applied to data paths of the form
    [frame,element,coords]
temporal
    Temporal transformations to be applied to data paths of the form
    [element,frame,coords]
"""
from . import spatial
from . import temporal
from .compose import Compose
from .spatiotemporalpath import SpatioTemporalPath
