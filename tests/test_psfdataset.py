# -----------------------------------------------------------
# Test PSFDataset class.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np
import os
import json

from psfdataset import PSFDataset
from psfdataset.transforms.spatial import Normalize


class TestPSFDataset:
    def test_DatasetAccess(self):
        # Test adding elements one by one and all access methods
        ds = PSFDataset(None)
        ds.add_element(np.array([1, 2]), 1)
        ds.add_element(np.array([2, 3]), 0)

        assert len(ds) == 2
        assert isinstance(ds[0], tuple)
        np.testing.assert_array_equal(ds[0][0], np.array([1, 2]))
        assert ds[0][1] == 1

        # Test iterator access
        i = 0
        for data, label in ds.get_iterator():
            np.testing.assert_array_equal(data, np.array([i + 1, i + 2]))
            assert label == 1 - i % 2
            i += 1

        assert ds.get_data_dimension() == 2

        assert isinstance(ds.get_description(), dict)

        def it():
            for i in range(2):
                yield (np.array([i + 1, i + 2]), 1 - i % 2)

        # Test creating from iterator
        ds2 = PSFDataset(None)
        ds2.fill_from_iterator(it())
        assert len(ds) == 2
        for i in range(2):
            np.testing.assert_array_equal(ds[i][0], np.array([i + 1, i + 2]))
            assert ds[i][1] == 1 - i % 2

    def test_SaveLoad(self):
        transform = Normalize(3, 1)
        ds = PSFDataset(transform)
        ds._data = [np.array([1, 2]), np.array([2, 3])]
        ds._labels = [1, 0]
        # test saving
        ds.save("test_ds")
        assert os.path.exists("test_ds.npz")
        assert os.path.exists("test_ds.json")
        # test loading
        ds2 = PSFDataset(transform)
        ds2.load("test_ds")
        assert len(ds) == len(ds2)
        for i in range(len(ds)):
            np.testing.assert_array_equal(ds[i][0], ds2[i][0])
            assert ds[i][1] == ds2[i][1]
        with open("test_ds.json", "r") as settings_file:
            transform_desc = json.load(settings_file)
        assert transform_desc == transform.get_description()
        # delete files
        os.remove("test_ds.npz")
        os.remove("test_ds.json")
