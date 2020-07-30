# -----------------------------------------------------------
# Test taking tuples of landmarks.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from psfdataset.transforms.spatial import Tuples


class TestTuples:
    def test_tuples(self):
        # 1 frame, 5 landmarks, 1D
        test_input = np.array([[[1], [2], [3], [4], [5], [6]]])
        # Test pairs
        tup = Tuples(2)
        expected = [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4],
                    [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6],
                    [5, 6]]
        tuples = tup(test_input)
        # 1 frame, 15 tuples (5 choose 2), 2 landmarks, 1D
        assert tuples.shape == (1, 15, 2, 1)
        for i in range(tuples.shape[1]):
            arr = list(np.sort(tuples[0, i].reshape(2)))
            assert arr in expected

        # Test triples
        tup = Tuples(3)
        expected = [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 3, 4],
                    [1, 3, 5], [1, 3, 6], [1, 4, 5], [1, 4, 6], [1, 5, 6],
                    [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 4, 5], [2, 4, 6],
                    [2, 5, 6], [3, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6]]
        tuples = tup(test_input)
        # 1 frame, 20 tuples (5 choose 3), 3 landmarks, 1D
        assert tuples.shape == (1, 20, 3, 1)
        for i in range(tuples.shape[1]):
            arr = list(np.sort(tuples[0, i].reshape(3)))
            assert arr in expected

        # test 4-tuples
        tup = Tuples(4)
        expected = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 4, 5],
                    [1, 2, 4, 6], [1, 2, 5, 6], [1, 3, 4, 5], [1, 3, 4, 6],
                    [1, 3, 5, 6], [1, 4, 5, 6], [2, 3, 4, 5], [2, 3, 4, 6],
                    [2, 3, 5, 6], [2, 4, 5, 6], [3, 4, 5, 6]]
        tuples = tup(test_input)
        # 1 frame, 15 tuples (5 choose 4), 4 landmarks, 1D
        assert tuples.shape == (1, 15, 4, 1)
        for i in range(tuples.shape[1]):
            arr = list(np.sort(tuples[0, i].reshape(4)))
            assert arr in expected
        assert isinstance(tup.get_description(), dict)
