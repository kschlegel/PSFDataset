# -----------------------------------------------------------
# Test time incorporated and invisibility reset transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.temporal import TimeIncorporatedTransformation
from psfdataset.transforms.temporal import InvisibilityResetTransformation


class TestPathTransformations:
    def test_TimeIncorporatedTransform(self):
        test_input = np.array([[[2], [4], [8]]])
        ti = TimeIncorporatedTransformation()
        expected = np.array([[[2, 0], [4, 0.5], [8, 1]]])
        output = ti(test_input)
        np.testing.assert_allclose(output, expected)
        assert isinstance(ti.get_description(), dict)

    def test_InvisibilityResetTransform(self):
        test_input = np.array([[[1], [2], [3]]])
        ir = InvisibilityResetTransformation()
        expected = np.array([[[1, 1], [2, 1], [3, 1], [3, 0], [0, 0]]])
        output = ir(test_input)
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == test_input.dtype
        assert isinstance(ir.get_description(), dict)
