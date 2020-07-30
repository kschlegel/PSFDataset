# -----------------------------------------------------------
# Test normalization transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.spatial import Normalize
from psfdataset.transforms.spatial import NormalizeWithConfidence
from psfdataset.transforms.spatial import NormalizeWithoutConfidence


class TestNormalize:
    def test_Normalize(self):
        # Test normalizing of pure spatial coordinate arrays
        # Test normalizing with given fixed data range boundarys
        test_input = np.array([[[1, 2, 3]]])
        normalize = Normalize(3, 1)
        expected = np.array([[[-1, 0, 1]]])
        output = normalize(test_input)
        np.testing.assert_array_equal(output, expected)

        # Test inferring data range from given element (results in varying data
        # range between different calls)
        test_input2 = np.array([[[2, 4, 10]]])
        normalize = Normalize()
        expected2 = np.array([[[-1, -0.5, 1]]])
        output = normalize(test_input)
        np.testing.assert_array_equal(output, expected)
        output = normalize(test_input2)
        np.testing.assert_array_equal(output, expected2)
        assert isinstance(normalize.get_description(), dict)

    def test_NormalizeConfidenceScores(self):
        # Test normalizing of arrays containing a confidence score as their
        # last dimension
        test_input = np.array([[[1, 0], [2, 0.5], [3, 1]]])
        # Test normalizing spatial coords, leaving the confidence untouched
        normalize = NormalizeWithoutConfidence(3, 1)
        expected = np.array([[[-1, 0], [0, 0.5], [1, 1]]])
        output = normalize(test_input)
        np.testing.assert_array_equal(output, expected)
        assert isinstance(normalize.get_description(), dict)

        # Test normalizing everything (confidence needs different scaling)
        normalize = NormalizeWithConfidence(3, 1)
        expected = np.array([[[-1, -1], [0, 0], [1, 1]]])
        output = normalize(test_input)
        np.testing.assert_array_equal(output, expected)
        assert isinstance(normalize.get_description(), dict)
