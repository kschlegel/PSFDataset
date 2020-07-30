# -----------------------------------------------------------
# Test composing transformations.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms import Compose
from psfdataset.transforms.spatial import Crop, Normalize


class TestCompose:
    def test_Compose(self):
        crop = Crop()
        norm = Normalize(2)
        transform = Compose([crop, norm])

        # test composition
        test_input = np.array([[[1, 2], [2, 4]]])
        expected = np.array([[[-1, -1], [0, 1]]])
        output = transform(test_input)
        np.testing.assert_array_equal(output, expected)

        # check desc array
        desc = transform.get_description()
        assert isinstance(desc, dict)
        for key, val in crop.get_description().items():
            assert key in desc
            assert desc[key] == val
        for key, val in norm.get_description().items():
            assert key in desc
            assert desc[key] == val

        assert "compose" in desc
        assert desc["compose"] == "(s)Crop->(s)Normalize"
