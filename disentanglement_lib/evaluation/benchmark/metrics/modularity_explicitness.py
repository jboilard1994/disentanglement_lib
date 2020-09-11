# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modularity and explicitness metrics from the F-statistic paper.

Based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss"
(https://arxiv.org/pdf/1802.05312.pdf).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.benchmark.metrics import utils
import numpy as np
from six.moves import range
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import gin.tf


@gin.configurable(
    "modularity_explicitness",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_modularity_explicitness(dataholder,
                                    random_state,
                                    artifact_dir=None,
                                    num_train=gin.REQUIRED,
                                    num_test=gin.REQUIRED,
                                    batch_size=16):
    """Computes the modularity metric according to Sec 3.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      num_train: Number of points used for training.
      num_test: Number of points used for testing.
      batch_size: Batch size for sampling.

    Returns:
      Dictionary with average modularity score and average explicitness
        (train and test).
    """
    del artifact_dir
    scores = {}
    mus_train, ys_train = utils.generate_batch_factor_code(dataholder, num_train, random_state, batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(dataholder, num_test, random_state, batch_size)

    all_mus = np.transpose(dataholder.embed_codes)
    all_ys = np.transpose(dataholder.factors)

    #New score
    discretized_mus, bins0 = utils.make_discretizer(all_mus, dataholder.cumulative_dist)
    mutual_information = utils.discrete_mutual_info(discretized_mus, all_ys)
    # Mutual information should have shape [num_codes, num_factors].
    assert mutual_information.shape[0] == mus_train.shape[0]
    assert mutual_information.shape[1] == ys_train.shape[0]
    scores["MODEX_modularity_score"] = modularity(mutual_information)

    # From paper : "for modularity, we report the mean across validation splits and embedding dimensions."
    # old implementation of disentanglement-lib used train set.
    # So we get results for whole dataset, train-set, and test set.

    #old score 1
    discretized_mus, bins1 = utils.make_discretizer(mus_train, dataholder.cumulative_dist)
    mutual_information = utils.discrete_mutual_info(discretized_mus, ys_train)
    scores["MODEX_modularity_oldtrain_score"] = modularity(mutual_information)

    #old score 2
    discretized_mus, bins2 = utils.make_discretizer(mus_test, dataholder.cumulative_dist)
    mutual_information = utils.discrete_mutual_info(discretized_mus, ys_test)
    scores["MODEX_modularity_oldtest_score"] = modularity(mutual_information)

    explicitness_score_train = np.zeros([ys_train.shape[0], 1])
    explicitness_score_test = np.zeros([ys_test.shape[0], 1])
    mus_train_norm, mean_mus, stddev_mus = utils.normalize_data(mus_train)
    mus_test_norm, _, _ = utils.normalize_data(mus_test, mean_mus, stddev_mus)
    for i in range(ys_train.shape[0]):
        explicitness_score_train[i], explicitness_score_test[i] = \
            explicitness_per_factor(random_state, mus_train_norm, ys_train[i, :],
                                    mus_test_norm, ys_test[i, :])
    scores["MODEX_explicitness_score_train"] = np.mean(explicitness_score_train)
    scores["MODEX_explicitness_score_test"] = np.mean(explicitness_score_test)
    return scores


def explicitness_per_factor(random_state, mus_train, y_train, mus_test, y_test):
    """Compute explicitness score for a factor as ROC-AUC of a classifier.

    Args:
      random_state : Seeds the logistic regressor
      mus_train: Representation for training, (num_codes, num_points)-np array.
      y_train: Ground truth factors for training, (num_factors, num_points)-np
        array.
      mus_test: Representation for testing, (num_codes, num_points)-np array.
      y_test: Ground truth factors for testing, (num_factors, num_points)-np
        array.

    Returns:
      roc_train: ROC-AUC score of the classifier on training data.
      roc_test: ROC-AUC score of the classifier on testing data.
    """
    x_train = np.transpose(mus_train)
    x_test = np.transpose(mus_test)
    clf = LogisticRegression(max_iter=500, random_state=random_state).fit(x_train, y_train)
    y_pred_train = clf.predict_proba(x_train)
    y_pred_test = clf.predict_proba(x_test)
    mlb = MultiLabelBinarizer()
    roc_train = roc_auc_score(mlb.fit_transform(np.expand_dims(y_train, 1)),
                              y_pred_train)
    roc_test = roc_auc_score(mlb.fit_transform(np.expand_dims(y_test, 1)),
                             y_pred_test)
    return np.abs(roc_train-0.5)*2, np.abs(roc_test-0.5)*2


def modularity(mutual_information):
    """Computes the modularity from mutual information."""
    # Mutual information has shape [num_codes, num_factors].
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] -1.)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)
