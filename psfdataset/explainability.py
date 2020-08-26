from typing import List
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

from .types import KeypointTransformation


class ExplainPSF:
    def __init__(self, landmarks: List[str], coordinates: List[str],
                 classes: List[str],
                 transform: KeypointTransformation) -> None:
        """

        """
        self._landmarks = landmarks
        self._coordinates = coordinates
        self._classes = classes

        # let the transform rework the input format and give back my featureset
        # structure
        output_structure = transform.explain(["time", landmarks, coordinates])
        if len(output_structure) != 3:
            raise Exception(
                "Currently the output is assumed to have exactly 3 dimensions."
            )
        if isinstance(output_structure[0], list):
            self._input_elements = output_structure[0]
        else:
            raise Exception("Input elements are not named.")

        if isinstance(output_structure[1], list):
            self._feature_groups = output_structure[1]
        else:
            raise Exception("The feature groups are not named.")
        if isinstance(output_structure[2], list):
            self._feature_group_size = len(output_structure[2])
        elif isinstance(output_structure[2], int):
            self._feature_group_size = output_structure[2]
        else:
            raise Exception("Feature groups must be named or define the "
                            "number of elements contained")
        self._feature_block = (len(self._feature_groups) *
                               self._feature_group_size)

        self._featureset__avg = None
        self._featureset__max = None

    def add_featureset(self, featureset):
        num_elements = len(self._input_elements)
        class_cnt_abs = [0] * len(self._classes)
        class_cnt_pos = np.zeros(
            (len(self._classes), num_elements * self._feature_block))
        class_cnt_neg = np.zeros(
            (len(self._classes), num_elements * self._feature_block))
        acc_abs_feat = np.zeros(
            (len(self._classes), num_elements * self._feature_block))
        acc_pos_feat = np.zeros(
            (len(self._classes), num_elements * self._feature_block))
        acc_neg_feat = np.zeros(
            (len(self._classes), num_elements * self._feature_block))
        for feature_vec, label in featureset.get_iterator():
            acc_abs_feat[label] += np.abs(feature_vec)
            class_cnt_abs[label] += 1

            pos_feat = feature_vec.copy()
            pos_feat[pos_feat < 0] = 0
            acc_pos_feat[label] += pos_feat
            class_cnt_pos[label] += (pos_feat != 0).astype(np.int32)

            neg_feat = feature_vec.copy()
            neg_feat[neg_feat > 0] = 0
            acc_neg_feat[label] += np.abs(neg_feat)
            class_cnt_neg[label] += (neg_feat != 0).astype(np.int32)

        class_cnt_pos[class_cnt_pos == 0] = 1
        class_cnt_neg[class_cnt_neg == 0] = 1

        for i in range(len(self._classes)):
            acc_abs_feat[i] /= class_cnt_abs[i]
            acc_pos_feat[i] /= class_cnt_pos[i]
            acc_neg_feat[i] /= class_cnt_neg[i]

        self._featureset_avg = {}
        self._featureset_avg["abs"] = acc_abs_feat
        self._featureset_avg["pos"] = acc_pos_feat
        self._featureset_avg["neg"] = acc_neg_feat
        self._featureset_max = {}
        self._featureset_max["abs"] = np.amax(self._featureset_avg["abs"])
        self._featureset_max["pos"] = np.amax(self._featureset_avg["pos"])
        self._featureset_max["neg"] = np.amax(self._featureset_avg["neg"])

    def action_class(self, mat, action_class):
        title = self._classes[action_class]
        ticks = self._input_elements
        mat_max = np.amax(mat)
        mat_min = np.amin(mat)
        sub_mat = mat[action_class].reshape(len(self._input_elements),
                                            self._feature_block)
        sub_featuresets = {}
        for mode in ("abs", "pos", "neg"):
            sub_featuresets[mode] = self._featureset_avg[mode][
                action_class].reshape(len(self._input_elements),
                                      self._feature_block)
        return self._single_item(title, ticks, sub_mat, mat_min, mat_max,
                                 sub_featuresets)

    def input_element(self, mat, input_element):
        title = self._input_elements[input_element]
        ticks = self._classes
        mat_max = np.amax(mat)
        mat_min = np.amin(mat)
        sub_mat = mat[:, self._feature_block *
                      input_element:self._feature_block * (input_element + 1)]
        sub_featuresets = {}
        for mode in ("abs", "pos", "neg"):
            sub_featuresets[mode] = self._featureset_avg[
                mode][:, self._feature_block *
                      input_element:self._feature_block * (input_element + 1)]
        return self._single_item(title, ticks, sub_mat, mat_min, mat_max,
                                 sub_featuresets)

    def _single_item(self, title, ticks, sub_mat, mat_min, mat_max,
                     sub_featuresets):
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(title, fontsize=16)
        bar_cnt = 0
        with tqdm(total=6, desc="Rendering views") as pbar:
            ax = plt.subplot(3, 2, 1)
            ax = sns.heatmap(sub_mat,
                             vmin=mat_min,
                             vmax=mat_max,
                             center=0,
                             yticklabels=ticks)
            ax.set_title("Linear map")
            pbar.update(bar_cnt)
            bar_cnt += 1

            for pos, mode in ((5, "abs"), (4, "pos"), (6, "neg")):
                ax = plt.subplot(3, 2, pos)
                ax = sns.heatmap(sub_featuresets[mode],
                                 vmin=0,
                                 vmax=self._featureset_max[mode],
                                 yticklabels=ticks)
                ax.set_title("Feature vector - " + mode)
                pbar.update(bar_cnt)
                bar_cnt += 1

            featureset_pos_neg = (sub_featuresets["pos"] -
                                  sub_featuresets["neg"])
            ax = plt.subplot(3, 2, 2)
            ax = sns.heatmap(featureset_pos_neg,
                             vmin=-self._featureset_max["neg"],
                             vmax=self._featureset_max["pos"],
                             center=0,
                             yticklabels=ticks)
            ax.set_title("Feature vector - pos-neg")
            pbar.update(bar_cnt)
            bar_cnt += 1

            ax = plt.subplot(3, 2, 3)
            ax = sns.heatmap(np.multiply(sub_mat, featureset_pos_neg),
                             center=0,
                             yticklabels=ticks)
            ax.set_title("Mat x Feature vector")
            pbar.update(bar_cnt)

        return fig

    def model_per_input_element(self, mat: np.ndarray) -> plt.figure:
        """
        Draw a heatmap representation of the given linear map.

        Splits the matrix up by input element, showing the importance of each
        given input element to all the possible classes.

        Parameters
        ----------
        mat: Numpy array
            Matrix representing the linear map learned for classification.
        self._feature_block: int
            Size of the section of the feature vector that corresponds to one
            input element.
        """
        rows = int(np.ceil(len(self._input_elements) / 2))

        fig = plt.figure(figsize=(20, 10 * rows))
        for i in trange(len(self._input_elements),
                        desc="Rendering input element plots"):
            ax = plt.subplot(rows, 2, i + 1)
            ax = sns.heatmap(mat[:, self._feature_block *
                                 i:self._feature_block * (i + 1)],
                             vmin=np.amin(mat),
                             vmax=np.amax(mat),
                             center=0,
                             yticklabels=self._classes)
            ax.set_title(self._input_elements[i])
        return fig

    def model_per_class(self, mat: np.ndarray) -> plt.figure:
        """
        Draw a heatmap representation of the given linear map.

        Splits the matrix up by class, showing the importance of all
        input elements to each given class.

        Parameters
        ----------
        mat: Numpy array
            Matrix representing the linear map learned for classification.
        self._feature_block: int
            Size of the section of the feature vector that corresponds to one
            input element.
        """
        rows = int(np.ceil(len(self._classes) / 2))

        fig = plt.figure(figsize=(20, 10 * rows))
        for i in trange(len(self._classes), desc="Rendering class plots"):
            ax = plt.subplot(rows, 2, i + 1)
            ax = sns.heatmap(mat[i].reshape(len(self._input_elements),
                                            self._feature_block),
                             vmin=np.amin(mat),
                             vmax=np.amax(mat),
                             center=0,
                             yticklabels=self._input_elements)
            ax.set_title(self._classes[i])
        return fig

    def featureset_per_class(self):
        if self._featureset_avg is None:
            raise Exception("The featureset neeads to be added using"
                            ".add_featureset() first.")

        rows = int(np.ceil(len(self._classes) / 2))
        tot_max = np.amax(self._featureset_avg["abs"])
        fig = plt.figure(figsize=(20, 10 * rows))
        for i in trange(len(self._classes), desc="Rendering class plots"):
            ax = plt.subplot(rows, 2, i + 1)
            ax = sns.heatmap(np.abs(self._featureset_avg["abs"][i].reshape(
                len(self._input_elements), self._feature_block)),
                             vmin=0,
                             vmax=tot_max,
                             yticklabels=self._input_elements)
            ax.set_title(self._classes[i])
        return fig

    def featureset_per_input_element(self):
        if self._featureset_avg is None:
            raise Exception("The featureset neeads to be added using"
                            ".add_featureset() first.")

        rows = int(np.ceil(len(self._input_elements) / 2))
        tot_max = np.amax(self._featureset_avg["abs"])
        fig = plt.figure(figsize=(20, 10 * rows))
        for i in trange(len(self._input_elements),
                        desc="rendering input element plots"):
            ax = plt.subplot(rows, 2, i + 1)
            ax = sns.heatmap(np.abs(
                self._featureset_avg["abs"][:, self._feature_block *
                                            i:self._feature_block * (i + 1)]),
                             vmin=0,
                             vmax=tot_max,
                             yticklabels=self._classes)
            ax.set_title(self._input_elements[i])
        return fig

    def save_all_classes(self, mat, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(os.path.join(folder, "classes")):
            os.mkdir(os.path.join(folder, "classes"))
        for i in trange(len(self._classes), desc="Class views"):
            fig = self.action_class(mat, i)
            plt.savefig(
                os.path.join(folder, "classes", self._classes[i] + ".png"))
            plt.close(fig)

    def save_all_input_elements(self, mat, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(os.path.join(folder, "elements")):
            os.mkdir(os.path.join(folder, "elements"))
        for i in trange(len(self._input_elements), desc="Input element views"):
            fig = self.input_element(mat, i)
            plt.savefig(
                os.path.join(folder, "elements",
                             self._input_elements[i] + ".png"))
            plt.close(fig)

    def input_element_vs_class(self, mat, input_element, action_class):
        fig = plt.figure(figsize=(20, 30))
        fig.suptitle(self._input_elements[input_element] + " - " +
                     self._classes[action_class],
                     fontsize=16)
        bar_cnt = 0
        mat_max = np.amax(mat)
        mat_min = np.amin(mat)

        featureset_pos_neg = (
            self._featureset_avg["pos"][:, self._feature_block *
                                        input_element:self._feature_block *
                                        (input_element + 1)] -
            self._featureset_avg["neg"][:, self._feature_block *
                                        input_element:self._feature_block *
                                        (input_element + 1)])

        with tqdm(total=8, desc="Rendering views") as pbar:
            ax = plt.subplot(4, 2, 1)
            ax = sns.heatmap(mat[action_class].reshape(
                len(self._input_elements), self._feature_block),
                             vmin=mat_min,
                             vmax=mat_max,
                             center=0,
                             yticklabels=self._input_elements)
            ax.set_title("Linear map - " + self._classes[action_class])
            pbar.update(bar_cnt)
            bar_cnt += 1

            ax = plt.subplot(4, 2, 2)
            ax = sns.heatmap(
                mat[:, self._feature_block *
                    input_element:self._feature_block * (input_element + 1)],
                vmin=mat_min,
                vmax=mat_max,
                center=0,
                yticklabels=self._classes)
            ax.set_title("Linear map - " + self._input_elements[input_element])
            pbar.update(bar_cnt)
            bar_cnt += 1

            ax = plt.subplot(4, 2, 3)
            ax = sns.heatmap(np.multiply(
                mat[action_class].reshape(len(self._input_elements),
                                          self._feature_block),
                self._featureset_avg["pos"][action_class].reshape(
                    len(self._input_elements), self._feature_block) -
                self._featureset_avg["neg"][action_class].reshape(
                    len(self._input_elements), self._feature_block)),
                             center=0,
                             yticklabels=self._input_elements)
            ax.set_title("Mat x Feature vector - " +
                         self._classes[action_class])
            pbar.update(bar_cnt)
            bar_cnt += 1

            ax = plt.subplot(4, 2, 4)
            ax = sns.heatmap(np.multiply(
                mat[:, self._feature_block *
                    input_element:self._feature_block * (input_element + 1)],
                featureset_pos_neg),
                             center=0,
                             yticklabels=self._classes)
            ax.set_title("Mat x Feature vector - " +
                         self._input_elements[input_element])
            pbar.update(bar_cnt)
            bar_cnt += 1

            for pos, mode in ((7, "abs"), (6, "pos"), (8, "neg")):
                ax = plt.subplot(4, 2, pos)
                ax = sns.heatmap(self._featureset_avg[mode]
                                 [:, self._feature_block *
                                  input_element:self._feature_block *
                                  (input_element + 1)],
                                 vmin=0,
                                 vmax=self._featureset_max[mode],
                                 yticklabels=self._classes)
                ax.set_title("Feature vector - " + mode)
                pbar.update(bar_cnt)
                bar_cnt += 1

            ax = plt.subplot(4, 2, 5)
            ax = sns.heatmap(featureset_pos_neg,
                             vmin=-self._featureset_max["neg"],
                             vmax=self._featureset_max["pos"],
                             center=0,
                             yticklabels=self._classes)
            ax.set_title("Feature vector - pos-neg")
            pbar.update(bar_cnt)

        return fig
