# -----------------------------------------------------------
# Test transforming spatial into spatiotemporal paths.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms import SpatioTemporalPath


class TestSpatioTemporalPath:
    def test_SpatioTemporalPath(self):
        test_input = np.array([[[-1, 0], [2, 3]], [[0, 1], [3, 4]],
                               [[1, -1], [4, 2]]])

        # Test with disintegrating the path
        transform = SpatioTemporalPath(disintegrate=True)
        exp = np.array([[[-1], [0], [1]], [[0], [1], [-1]], [[2], [3], [4]],
                        [[3], [4], [2]]])
        output = transform(test_input)
        np.testing.assert_array_equal(output, exp)
        assert output.dtype == test_input.dtype

        # Test without disintegrating the path
        transform = SpatioTemporalPath(disintegrate=False)
        expected = np.array([[[-1, 0], [0, 1], [1, -1]],
                             [[2, 3], [3, 4], [4, 2]]])
        output = transform(test_input)
        np.testing.assert_array_equal(output, expected)

        # Test data that needs to be flattened from 4D to 3D
        test_input = np.array([[[[-1, 0], [2, 3]], [[0, -1], [3, 2]]],
                               [[[0, 1], [3, 4]], [[1, 0], [4, 3]]],
                               [[[1, -1], [4, 2]], [[-1, 1], [2, 4]]]])
        expected = np.array([[[-1, 0, 2, 3], [0, 1, 3, 4], [1, -1, 4, 2]],
                             [[0, -1, 3, 2], [1, 0, 4, 3], [-1, 1, 2, 4]]])
        output = transform(test_input)
        np.testing.assert_array_equal(output, expected)
        assert isinstance(transform.get_description(), dict)
