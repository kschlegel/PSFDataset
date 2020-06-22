# -----------------------------------------------------------
# Class to create and handle Path-Signature-Feature Datasets.
#
# (C) 2020 Kevin Schlegel, Oxford, United Kingdom
# Released under Apache License, Version 2.0
# email kevinschlegel@cantab.net
# -----------------------------------------------------------
import numpy as np
import json
from tqdm import tqdm

from .psfdatasubset import PSFDataSubset


class PSFDataset:
    """
    A class to create and handle Path-Signature-Feature Datasets.

    This class aims to provide a convient interface for creating datasets using
    pathsignature features for human action recognition from landmark data
    (see https://arxiv.org/abs/1707.03993).
    Datasets keep track of the transformations applied to the data during
    creation. Transformations are callable objects, inspired by torchvision
    transformations. They can be chained using Compose.
    Datasets can be saved to file, one npz file containing data and labels only
    (indexed as 'data' and 'labels') is created for easy reading in elsewhere.
    Another json file is created containing the properties of the dataset such
    as transformations that where applied to the data.

    Methods
    -------
    add_element(keypoints, label)
        Take keypoints and label and add them to the dataset.
    set_split(desc, train_ids, test_ids)
        Sets the training/test split for the dataset.
    from_iterator(data_iterator)
        Take iterator for keypoint label pairs and fill dataset.
    get_iterator()
        Python generator for iteration over dataset.
    get_data_dimension()
        Return dimension of feature vector.
    get_labels()
        Return a numpy array of all labels of the dataset.
    get_desc()
        Return a dictionary describing the properties of the dataset.
    save(filename)
        Save the dataset.
    load(filename)
        Load the dataset.
    """
    def __init__(self, transform=None, flattened=True, dtype=np.float64):
        """
        Parameters
        ----------
        transform : callable object from the transforms subpackage, optional
            A transformation to be applied to every new data element
            added to the dataset. (default is None)
        """
        # _data contains the (transformed) keypoint data
        self._data = []
        # _labels contains the ground truth classification labels
        self._labels = []
        # _transform is a callable sequence of transformations to be applied to
        # every data element when added
        self._transform = transform
        self._flattened = flattened
        self._dtype = dtype
        # optionally the dataset can hold a training/testset split
        # using the PSFDataSubset module
        self._trainingset = None
        self._testset = None
        self._split_desc = {}

    # properties to access trainingset and testset subsets of the dataset.
    # No setters implemented as setting the subsets should only happen through
    # the set_split method which also sets the description dictionary.
    @property
    def trainingset(self):
        """
        Access to the training subset of the dataset.
        """
        return self._trainingset

    @property
    def testset(self):
        """
        Access to the test subset of the dataset.
        """
        return self._testset

    def __getitem__(self, index):
        """ Returns the flattened feature vector and its label. """
        if self._flattened:
            return (self._data[index].reshape(-1), self._labels[index])
        else:
            return (self._data[index], self._labels[index])

    def __len__(self):
        return len(self._data)

    def add_element(self, keypoints, label):
        """
        Takes keypoints and label and add them to the dataset.

        Takes a numpy array of keypoints of the form [frame_id,keypoint,coords]
        and applies the transformation to the keypoints. Adds the transformed
        keypoints and the target label to the dataset.

        Parameters:
        -----------
        keypoints : numpy array of keypoints
            Original landmark data to be transformed and added to the dataset.
        label: int
            Ground truth classification label
        """
        if self._transform is not None:
            keypoints = self._transform(keypoints)
        self._data.append(keypoints.astype(self._dtype))
        self._labels.append(label)

    def set_split(self, desc, train_ids, test_ids):
        """
        Sets the training/test split for the dataset.

        The subsets can then be accessed via the trainingset and testset
        properties.

        Parameters
        ----------
        desc : dict
            Dictionary with all information to identify the split in the logs.
        train_ids : list of ints
            List of the ids of elements of the trainingset
        test_ids : list of ints
            List of the ids of elements of the testset
        """
        self._split_desc = desc
        self._trainingset = PSFDataSubset(self, train_ids)
        self._testset = PSFDataSubset(self, test_ids)

    def from_iterator(self, data_iterator):
        """
        Fill dataset with data using given iterator.

        Takes an iterator returning a list or tuple of keypoints (of shape
        [frame_id,keypoint,coords]),label pairs and adds everything to the
        dataset using the add_element method.

        Parameters
        ----------
        data_iterator: iterable
            Iterable returning keypoint,label pairs of data.
        """
        for element in tqdm(data_iterator):
            self.add_element(element[0], element[1])

    def get_iterator(self):
        """ Python generator for iterating over the dataset. """
        for i in range(len(self._data)):
            yield self[i]

    def get_data_dimension(self):
        """
        Returns size of feature vector.

        Returns the size of the flattened array of one dataset entry for
        determining the input size of a model.

        Returns
        -------
        int
            The size of the feature vector
        """
        if len(self._data) > 0:
            if self._flattened:
                return np.prod(self._data[0].shape)
            else:
                return self._data[0].shape
        else:
            return 0

    def get_labels(self):
        """
        Return array of all labels of the entire dataset.

        This is useful e.g. for computation of metrics after training epochs.

        Returns
        -------
        numpy array
            Labels of the dataset
        """
        return np.array(self._labels)

    def get_desc(self):
        """
        Returns a dictionary describing all properties of the dataset.

        The dictionary helps to keep track of the properties of the dataset. It
        also gets written to file when saving the dataset.
        The dict can also be passed into TensorBoard for hparam tracking.

        Currently the dict contains only the transformations applied.

        Returns
        -------
        dict
            Description of the dataset
        """
        if self._transform:
            desc = self._transform.get_desc()
        else:
            desc = {}
        desc.update(self._split_desc)
        return desc

    def save(self, filename):
        """
        Saves the dataset to file.

        Saves the data and labels to an .npz file for easy loading anywhere.
        Data and labels are indexed as 'data' and 'labels' repsectively in
        the .npz file.
        Saves the settings used to create the dataset into a second file to
        allow easy keeping track of its properties. Reloading transformations
        is currently not supported.

        Parameters
        ----------
        filename : string
            full filepath and filename without an extension (added
                automatically for the two files)
        """
        np.savez(filename,
                 data=np.array(self._data),
                 labels=np.array(self._labels))
        transform = self._transform.get_desc()
        with open(filename + ".json", "w") as json_file:
            json.dump(transform, json_file)

    def load(self, filename):
        """
        Load the dataset from file.

        Loads the data file created by the save method to load data and
        labels. Loading of settings of the dataset is currently not supported.

        Parameters
        ----------
        filename: string
            full filepath and filename without an extension (added
                automatically for the two files)
        """
        with np.load(filename + ".npz") as data:
            self._data = data['data']
            self._labels = data['labels']
