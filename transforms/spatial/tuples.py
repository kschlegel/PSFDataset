# -----------------------------------------------------------
# Class to form all (unordered) tuples of given size of landmarks.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np
from functools import partial


class Tuples:
    """
    From all (unordered) n-tuples of landmarks.

    Takes an array of the form [frame][landmark][coords] and for every frame
    forms every possible (unordered) n-tuples of landmarks. Returns an array
    [frame][tuple][landmark][coords] of n-tuples of landmarks.

    Methods
    -------
    get_desc()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, tuple_size):
        """
        Parameters
        ----------
        tuple_size : int
            size of tuples to be formed
        """
        self._tuple_size = tuple_size
        # Tuples and triples are going to be a most common
        # -> use an easier implementation
        if tuple_size == 2:
            self._tuple_fn = self._pairs
        elif tuple_size == 3:
            self._tuple_fn = self._triples
        else:
            self._tuple_fn = partial(self._Tuple, m=tuple_size)

    def __call__(self, sample):
        tuples = []
        for frame in range(sample.shape[0]):
            tuples_frame = []
            for tup in self._tuple_fn(n=sample.shape[1]):
                tuples_frame.append(sample[frame][tup])
            tuples.append(tuples_frame)
        return np.array(tuples)

    def get_desc(self):
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(SP)Tup": self._tuple_size}

    def _pairs(self, n):
        """
        Generator for all unordered pairs of indices up to n.
        """
        for i in range(n):
            for j in range(i + 1, n):
                yield [i, j]

    def _triples(self, n):
        """
        Generator for all unordered triples of indices up to n.
        """
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    yield [i, j, k]

    class _Tuple:
        """
        Generator for all unordered m-tuples of indices up to n.
        """
        def __init__(self, m, n):
            self.m = m
            self.n = n
            self.counter = [i for i in range(m)]
            self.counter[m - 1] -= 1

        def __iter__(self):
            return self

        def __next__(self):
            index = self.m - 1
            while (self.counter[index] == (self.n - (self.m - index))):
                index -= 1
                if index == -1:
                    raise StopIteration
            self.counter[index] += 1
            for i in range(1, self.m - index):
                self.counter[index + i] = self.counter[index + i - 1] + 1
            return self.counter
