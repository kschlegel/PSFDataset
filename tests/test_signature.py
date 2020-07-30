# -----------------------------------------------------------
# Test signature transform for tuples.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.spatial import Signature


class TestSignature:
    def test_Signature(self):
        # esig produces garbage if not fed with doubles. Feeding it
        # integers here makes sure they get converted before going into esig.
        test_input = np.array([[[[1, 1], [2, 2]]]])
        sig_lvl = 2
        sig = Signature(sig_lvl, False)
        expected = np.array([[[1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]]])
        output = sig(test_input)
        np.testing.assert_array_equal(output, expected)

        # Test dropping the zeroth term and a higher signature level
        sig_lvl = 3
        sig = Signature(sig_lvl, True)
        # The first 1.0 in the signature is omitted here to check drop_zero
        # The full signature would be [1,1,1,0.5,....]
        expected = np.array([[[
            1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.16666667, 0.16666667, 0.16666667,
            0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667
        ]]])
        output = sig(test_input)
        np.testing.assert_allclose(output, expected)
        assert isinstance(sig.get_description(), dict)
