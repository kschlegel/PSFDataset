# -----------------------------------------------------------
# Test lead lag and multi-delayed transformations
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.temporal import LeadLagTransformation
from psfdataset.transforms.temporal import MultiDelayedTransformation


class TestLeadLag:
    def test_LeadLagTransformation(self):
        test_input = np.array([[[1, 2], [2, 3], [3, 4], [4, 5]]])
        # Test standard lead-lag transformation, delayed by 1 time step
        llt = LeadLagTransformation(1)
        expected = np.array([[[[1, 2], [1, 2]], [[2, 3], [1, 2]],
                              [[2, 3], [2, 3]], [[3, 4], [2, 3]],
                              [[3, 4], [3, 4]], [[4, 5], [3, 4]],
                              [[4, 5], [4, 5]]]])
        output = llt(test_input)
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == test_input.dtype

        # Test lead lag with bigger delay interval
        llt = LeadLagTransformation(2)
        expected = ([[[[1, 2], [1, 2], [1, 2]], [[2, 3], [1, 2], [1, 2]],
                      [[2, 3], [2, 3], [1, 2]], [[2, 3], [2, 3], [2, 3]],
                      [[3, 4], [2, 3], [2, 3]], [[3, 4], [3, 4], [2, 3]],
                      [[3, 4], [3, 4], [3, 4]], [[4, 5], [3, 4], [3, 4]],
                      [[4, 5], [4, 5], [3, 4]], [[4, 5], [4, 5], [4, 5]]]])
        output = llt(test_input)
        np.testing.assert_array_equal(output, expected)

        # Test a 1D input
        test_input = np.array([[[1], [2], [3]]])
        exp = np.array([[[[1], [1], [1]], [[2], [1], [1]], [[2], [2], [1]],
                         [[2], [2], [2]], [[3], [2], [2]], [[3], [3], [2]],
                         [[3], [3], [3]]]])
        output = llt(test_input)
        np.testing.assert_array_equal(output, exp)
        assert isinstance(llt.get_description(), dict)

    def test_MultiDelayedTransformation(self):
        test_input = np.array([[[1, 2], [2, 3], [3, 4], [4, 5]]])
        # Test the with just a single time step delay
        mdt = MultiDelayedTransformation(1)
        expected = np.array([[[[1, 2], [0, 0]], [[2, 3], [1, 2]],
                              [[3, 4], [2, 3]], [[4, 5], [3, 4]],
                              [[0, 0], [4, 5]]]])
        output = mdt(test_input)
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == test_input.dtype

        # Test with multiple time step delay
        mdt = MultiDelayedTransformation(2)
        exp = np.array([[[[1, 2], [0, 0], [0, 0]], [[2, 3], [1, 2], [0, 0]],
                         [[3, 4], [2, 3], [1, 2]], [[4, 5], [3, 4], [2, 3]],
                         [[0, 0], [4, 5], [3, 4]], [[0, 0], [0, 0], [4, 5]]]])
        output = mdt(test_input)
        np.testing.assert_array_equal(output, exp)
        assert isinstance(mdt.get_description(), dict)
