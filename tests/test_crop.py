# -----------------------------------------------------------
# Test crop transformation.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.spatial.crop import Crop


class TestCrop:
    def test_Crop(self):
        # test simple crop of everything
        test_input = np.array([[[1, 1], [2, 2]]])
        crop = Crop()
        expected = np.array([[[0, 0], [1, 1]]])
        output = crop(test_input)
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == test_input.dtype
        assert isinstance(crop.get_description(), dict)

    def test_CropConfidenceScores(self):
        # test crop with ignoring points with 0 confidence (missing data)
        # points with 0 confidence are considered missing and come out = 0
        test_input = np.array([[[1, 0], [2, 0.5], [3, 1]]])
        crop = Crop(ignore_missing=True)
        expected = np.array([[[0, 0], [0, 0.5], [1, 1]]])
        output = crop(test_input)
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == test_input.dtype
        assert isinstance(crop.get_description(), dict)
