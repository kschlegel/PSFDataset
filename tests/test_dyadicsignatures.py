# -----------------------------------------------------------
# Test dyadic signature transformation.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.temporal import DyadicPathSignatures


class TestDyadicSignature:
    def test_DyadicPathSignature(self):
        # Testing sequence of 4 frames (dyadic splitting for sequence of odd
        # length is ambigious).

        # Test without any dyadic splitting, just signature
        # 1 element, 4 frames, 2D
        test_input = np.array([[[1, 2], [2, 3], [4, 5], [7, 8]]])
        dy_lvl = 0
        sig_lvl = 2
        sig = DyadicPathSignatures(dyadic_levels=dy_lvl,
                                   signature_level=sig_lvl)
        # expected 1 element, 1 full piece, sigdim(2D, lvl2)
        expected = np.array([[[1.0, 6.0, 6.0, 18.0, 18.0, 18.0, 18.0]]])
        output = sig(test_input)
        np.testing.assert_array_equal(output, expected)

        # Test with 1 non-overlapping dyadic level (full interval+the 2 halfs)
        dy_lvl = 1
        sig_lvl = 3
        sig = DyadicPathSignatures(dyadic_levels=dy_lvl,
                                   signature_level=sig_lvl)
        # expected 1 element, 1 full piece+2 halfs, sigdim(2D, lvl2)
        expected = [None] * 4
        # signature of full piece
        expected[0] = np.array([
            1.0, 6.0, 6.0, 18.0, 18.0, 18.0, 18.0, 36.0, 36.0, 36.0, 36.0,
            36.0, 36.0, 36.0, 36.0
        ])
        # signature of first half
        expected[1] = np.array([
            1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.16666667, 0.16666667,
            0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
            0.16666667
        ])
        # signature of middle half
        expected[2] = np.array([
            1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.33333333, 1.33333333,
            1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333,
            1.33333333
        ])
        # signature of second half
        expected[3] = np.array([
            1.0, 3.0, 3.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
            4.5, 4.5
        ])
        expected_stack = np.stack((expected[0], expected[1], expected[3]),
                                  axis=0)
        expected_stack = expected_stack.reshape((1, ) + expected_stack.shape)
        output = sig(test_input)
        np.testing.assert_allclose(output, expected_stack)

        # Test with 1 overlapping dyadic level
        # (full interval + 3 halfs: start, middle, end)
        sig = DyadicPathSignatures(dyadic_levels=dy_lvl,
                                   signature_level=sig_lvl,
                                   overlapping=True)
        # expected 1 element, 1 full piece+3 halfs, sigdim(2D, lvl2)
        expected_stack = np.stack(expected, axis=0)
        expected_stack = expected_stack.reshape((1, ) + expected_stack.shape)
        output = sig(test_input)
        np.testing.assert_allclose(output, expected_stack)

        assert isinstance(sig.get_description(), dict)
