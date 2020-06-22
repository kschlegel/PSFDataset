# -----------------------------------------------------------
# Class to zip PSFDatasets, handling their extra structures.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np


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
    get_desc()
        Return a dictionary describing the properties of the dataset.

    """
    def __init__(self, datasets, flattened=True):
        """
        Parameters
        ----------
        datasets : list/tuple of PSFDatasets
            A collection of existing PSFDatasets to be joined into one dataset.
        """
        if not isinstance(datasets, (list, tuple)) or len(datasets) < 2:
            raise Exception("Zipping datasets requires at least 2 datasets!")
        self._datasets = datasets
        self._flattened = flattened

    def __getitem__(self, index):
        """ Returns the flattened feature vector and its label. """
        keypoint_arr = []
        label = None
        for dataset in self._datasets:
            keypoints, label = dataset[index]
            keypoint_arr.append(keypoints)
        if self._flattened:
            return (np.concatenate(keypoint_arr), label)
        else:
            return (tuple(keypoint_arr), label)

    def __len__(self):
        return min([len(d) for d in self._datasets])

    def get_iterator(self):
        """ Python generator for iterating over the dataset. """
        for i in range(len(self)):
            yield self[i]

    def get_data_dimension(self):
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

    def get_labels(self):
        """
        Return array of all labels of the entire dataset.

        This is useful e.g. for computation of metrics after training epochs.

        Returns
        -------
        numpy array
            Labels of the dataset
        """
        return self._datasets[-1].get_labels()

    def get_desc(self):
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
        desc = {}
        for i, dataset in enumerate(self._datasets):
            ds_desc = dataset.get_desc()
            for key, val in ds_desc.items():
                desc["[DS" + str(i + 1) + "]" + key] = val
        return desc
