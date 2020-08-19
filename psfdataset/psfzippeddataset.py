# -----------------------------------------------------------
# Class to zip PSFDatasets, handling their extra structures.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np
from typing import Sequence, Iterator, Union, Tuple, Any

from .types import KeypointLabelPair, DescriptionDict
from .psfdataset import PSFDataset


class PSFZippedDataset():
    """
    Standard zipping of datasets, including handling of PSFDataset structures.

    Implemented here rather than using existing zip methods in machine learning
    libraries to handle the extra structures of PSFDatasets such as the
    description array.

    Exposes the same access interface as PSFDataset, creating, saving and
    loading needs to be done on an individual dataset basis.

    Methods
    -------
    get_iterator()
        Python generator for iteration over dataset.
    get_data_dimension()
        Return dimension of feature vector.
    get_labels()
        Return a numpy array of all labels of the dataset.
    get_description()
        Return a dictionary describing the properties of the dataset.

    """
    def __init__(self,
                 datasets: Sequence[PSFDataset],
                 flattened: bool = True) -> None:
        """
        Parameters
        ----------
        datasets : Collection of PSFDatasets
            A collection of existing PSFDatasets to be joined into one dataset.
            Datasets are assumed to come from the same dataset, i.e. have the
            same length and consistent labels across the collection
        """
        if not isinstance(datasets, (list, tuple)) or len(datasets) < 2:
            raise Exception("Zipping datasets requires at least 2 datasets!")
        ds_iter = iter(datasets)
        ds_len = len(next(ds_iter))
        for ds in ds_iter:
            if len(ds) != ds_len:
                raise Exception(
                    "All datasets in the collection must have equal length.")
        self._datasets = datasets
        self._flattened = flattened

    def __getitem__(self, index: int) -> KeypointLabelPair:
        """ Returns the flattened feature vector and its label. """
        keypoint_arr = []
        for dataset in self._datasets:
            keypoints, label = dataset[index]
            keypoint_arr.append(keypoints)
        if self._flattened:
            return (np.concatenate(keypoint_arr), label)
        else:
            return (tuple(keypoint_arr), label)

    def __len__(self) -> int:
        return len(self._datasets[0])

    def get_iterator(self) -> Iterator[KeypointLabelPair]:
        """ Python generator for iterating over the dataset. """
        for i in range(len(self)):
            yield self[i]  # return self[i] to use __getitem__ implementation

    def get_data_dimension(self) -> Union[int, Tuple[Any, ...]]:
        """
        Returns size of feature vector.

        Returns the size of the flattened and concatenated array of all the
        datasets combined for determining the input size of a model.

        Returns
        -------
        int
            The size of the feature vector
        """
        if self._flattened:
            return np.sum([d.get_data_dimension() for d in self._datasets])
        else:
            return tuple([d.get_data_dimension() for d in self._datasets])

    def get_labels(self) -> np.ndarray:
        """
        Return array of all labels of the entire dataset.

        This is useful e.g. for computation of metrics after training epochs.

        Returns
        -------
        numpy array
            Labels of the dataset
        """
        return self._datasets[-1].get_labels()

    def get_description(self) -> DescriptionDict:
        """
        Returns a dictionary describing all properties of all datasets.

        Collates all description arrays of the zipped datasets, prefixing their
        keys with the dataset number.
        The dictionary helps to keep track of the properties of the dataset. It
        also gets written to file when saving the dataset.
        The dict can also be passed into TensorBoard for hparam tracking.

        Currently the dict contains only the transformations applied.

        Returns
        -------
        dict
            Description of the dataset
        """
        desc: DescriptionDict = {}
        for i, dataset in enumerate(self._datasets):
            ds_desc = dataset.get_description()
            for key, val in ds_desc.items():
                desc["[DS" + str(i + 1) + "]" + key] = val
        return desc
