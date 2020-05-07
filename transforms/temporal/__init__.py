# -----------------------------------------------------------
# Package init file for PSFDataset temporal transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
"""
This subpackage defines the various temporal path transformations needed to
create datasets using the path signature feature methodology

Classes
-------
TimeIncorporatedTransformation
    Computes the time incorporated transformation of the path.
InvisibilityResetTransformation
    Computes the invisibility reset transformation of the path.
LeadLagTransformation
    Computes the lead lag transformation of the path.
MultiDelayedTransformation
    Computes the multi-delayed transformation of the path.
DyadicPathSignatures
    Optionally splits the path into dyadic intervals and computes signatures of
    every piece.
"""

from .pathtransformations import TimeIncorporatedTransformation
from .pathtransformations import InvisibilityResetTransformation
from .leadlag import LeadLagTransformation, MultiDelayedTransformation
from .dyadicsignatures import DyadicPathSignatures
