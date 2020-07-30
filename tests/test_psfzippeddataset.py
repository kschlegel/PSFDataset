# -----------------------------------------------------------
# Test zipping psf datasets.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np
import pytest

from psfdataset import PSFDataset
from psfdataset import PSFZippedDataset
from psfdataset.transforms.spatial import Normalize, Crop


class TestPSFZippedDataset:
    def test_PSFZippedDataset(self):
        # Check zipping
        ds1 = PSFDataset(Crop())
        ds1.add_element(np.array([[[1], [2]]]), 1)
        ds1.add_element(np.array([[[2], [4]]]), 0)

        ds2 = PSFDataset(Normalize(3, 1))
        ds2.add_element(np.array([1, 2]), 1)  # 0 -> -1 0
        ds2.add_element(np.array([2, 3]), 0)  # 1 -> 0 1
        ds2.add_element(np.array([1, 3]), 0)  # 2-> -1 1

        with pytest.raises(Exception):
            ds = PSFZippedDataset(ds1)
            ds = PSFZippedDataset([ds1])
            ds = PSFZippedDataset((ds1, ds2))
        ds1.add_element(np.array([[[3], [6]]]), 0)
        ds = PSFZippedDataset((ds1, ds2))

        assert len(ds) == 3
        assert ds.get_data_dimension() == 4
        np.testing.assert_array_equal(ds[0][0], np.array([0, 1, -1, 0]))
        np.testing.assert_array_equal(ds[1][0], np.array([0, 2, 0, 1]))
        assert ds[0][1] == 1
        assert ds[1][1] == 0

        # Test iterator access
        i = 0
        for data, label in ds.get_iterator():
            np.testing.assert_array_equal(
                data, np.array([0, i + 1, i % 2 - 1,
                                int(i > 0)]))
            assert label == int(i == 0)
            i += 1

        # Check description array
        desc = ds.get_description()
        assert isinstance(desc, dict)
        for key, val in ds1.get_description().items():
            assert "[DS1]" + key in desc
            assert desc["[DS1]" + key] == val
        for key, val in ds2.get_description().items():
            assert "[DS2]" + key in desc
            assert desc["[DS2]" + key] == val
