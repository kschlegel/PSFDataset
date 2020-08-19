# -----------------------------------------------------------
# Class to allow easy access to subsets of the full dataset for more efficient
# cross validation.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np

from typing import List, Iterator

from .types import KeypointLabelPair, PSFDatasetObject


class PSFDataSubset:
    """
    Provides easy access to subsets of the full dataset.

    This class is to allow creating a single dataset object, holding all the
    data and provide easy access to split the dataset into subsets for train/
    test(/valid) splits.

    This can be used either as a wrapper around an existing PSF(Zipped)Dataset
    object and provides all the general access functions the normal dataset
    class provides. Pass in the original dataset (PSFDatset or
    PSFZippedDataset) and the ids belonging to the subset and access the data
    of the subset as if it was a PSF(Zipped)Dataset.
    Or it can be used integrated into PSFDataset or PSFZippedDataset, by
    setting using the set_split method. The subsets can then be accessed via
    the trainingset and testset properties of the dataset.

    Methods
    -------
    get_iterator()
        Python generator for iteration over the subset.
    get_labels()
        Return a numpy array of all labels of the subset.
    """
    def __init__(self, dataset: PSFDatasetObject, ids: List[int]) -> None:
        """
        Parameters
        ----------
        dataset : PSF(Zipped)Datset object
            The object holding the full dataset
        ids : list of ints
            List of the ids belonging to the given subset
        """
        self._dataset = dataset
        self._ids = ids

    def __getitem__(self, index: int) -> KeypointLabelPair:
        return self._dataset[self._ids[index]]

    def __len__(self) -> int:
        return len(self._ids)

    def get_iterator(self) -> Iterator[KeypointLabelPair]:
        """ Python generator for iterating over the subset. """
        for i in range(len(self._ids)):
            yield self[i]

    def get_labels(self) -> np.ndarray:
        """
        Return array of all labels of the subset.

        This is useful e.g. for computation of metrics after training epochs.

        Returns
        -------
        numpy array
            Labels of the dataset
        """
        return self._dataset.get_labels()[self._ids]
