# -----------------------------------------------------------
# Classes to take various kinds of subsets of frames from the original
# sequence.
# Either subsample every n-th frame to reduce the amount of data processed
# or take a random subset or the first n frames (zero-pad if necessary) to
# obtain fixed size input when needed.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np


def _zero_pad(sample, length):
    if len(sample) == length:
        return sample
    else:
        padding = np.zeros((length - len(sample), ) + sample.shape[1:])
        return np.concatenate([sample, padding])


class RandomSubset:
    """
    Extract a random subset of sample_size frames from the data.

    Randomly samples n frames from a given input, preserving their original
    order. If the input has fewer frames than to be sampled all frames are
    returned and the result is padded with zeros at the end.

    Methods
    -------
    get_desc()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, sample_size, seed=None):
        """
        Parameters
        ----------
        sample_size : int
            How many frames to sample
        seed: int, optional (default is None)
            seed to initialize the random number generator. Ignored if None.
        """
        self._sample_size = sample_size
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, sample):
        if len(sample) > self._sample_size:
            rnd_sample = []
            while len(rnd_sample) < self._sample_size:
                rnd = np.random.randint(0, len(sample))
                if rnd not in rnd_sample:
                    rnd_sample.append(rnd)
            return sample[np.sort(rnd_sample)]
        else:
            # input is shorter than sample size, return everything zero-padded
            return _zero_pad(sample, self._sample_size)

    def get_desc(self):
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        desc = {"(s)rnd/sample_size": self._sample_size}
        if self._seed is not None:
            desc["(s)rnd/seed"] = self._seed
        return desc


class FirstN:
    """
    Extract the first n frames from the data.

    Extracts n frames from a given input. If the input has fewer frames, then
    all frames are returned and the result is padded with zeros at the end.

    Methods
    -------
    get_desc()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            Number of frames to be extracted
        """
        self._n = n

    def __call__(self, sample):
        if len(sample) >= self._n:
            return sample[0:self._n]
        else:
            # input is shorter than sample size, return everything zero-padded
            return _zero_pad(sample, self._n)

    def get_desc(self):
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(s)FirstN": self._n}


class SubSample:
    """
    Extract a subsample of every n-th frame from the data.

    Extracts every n-th frame from the data, returning a variable length
    sequence of frames, depending on the size of the original input. Combine
    this with the FirstN transformation to obtain a fixed sample size.

    Methods
    -------
    get_desc()
        Return a dictionary describing the properties of the transformation.
    """
    def __init__(self, step_size):
        """
        Parameters
        ----------
        step_size : int
            sample every step_size frame from the data
        """
        self._step_size = step_size

    def __call__(self, sample):
        sub_sample = [i for i in range(0, sample.shape[0], self._step_size)]
        return sample[sub_sample]

    def get_desc(self):
        """
        Returns a dictionary describing all properties of the transformation.

        Returns
        -------
        dict
            Description of the transformation
        """
        return {"(s)subsample": self._step_size}
