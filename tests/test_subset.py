# -----------------------------------------------------------
# Test the various ways of taking subsets of the data.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.spatial import RandomSubset, FirstN, SubSample


class TestSubset:
    def test_RandomSubset(self):
        rnd_subset = RandomSubset(3)
        seq = np.array([i for i in range(5)])
        assert len(rnd_subset(seq)) == 3
        # Test padding with zeros
        seq = np.array([i for i in range(2, 4)])
        sub_seq = list(rnd_subset(seq))
        assert len(sub_seq) == 3
        assert sub_seq == [2, 3, 0] or sub_seq == [3, 2, 0]
        assert isinstance(rnd_subset.get_description(), dict)

    def test_FirstN(self):
        first_n = FirstN(3)
        seq = np.array([i for i in range(5)])
        assert list(first_n(seq)) == [0, 1, 2]
        # Test padding with zeros
        seq = np.array([i for i in range(2)])
        assert list(first_n(seq)) == [0, 1, 0]
        assert isinstance(first_n.get_description(), dict)

    def test_SubSample(self):
        test_input = np.array([i for i in range(30)])
        subsample = SubSample(5)
        expected = np.array([0, 5, 10, 15, 20, 25])
        output = subsample(test_input)
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == test_input.dtype
        assert isinstance(subsample.get_description(), dict)
