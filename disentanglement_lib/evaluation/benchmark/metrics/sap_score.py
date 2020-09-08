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

"""Implementation of the SAP score.

Based on "Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations" (https://openreview.net/forum?id=H1kG7GZAW), Section 3.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.benchmark.metrics import utils
import numpy as np
from six.moves import range
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import gin.tf


@gin.configurable(
    "sap_score",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_sap(dataholder,
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                continuous_factors=gin.REQUIRED):
    """Computes the SAP score.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing discrete variables.
    batch_size: Batch size for sampling.
    continuous_factors: Factors are continuous variable (True) or not (False).

    Returns:
    Dictionary with SAP score.
    """
    del artifact_dir
    logging.info("Generating training set.")
    mus, ys = utils.generate_batch_factor_code(dataholder, num_train, random_state, num_train, continuous=continuous_factors)
    mus_test, ys_test = utils.generate_batch_factor_code(dataholder, num_test, random_state, num_test, continuous=continuous_factors)
    logging.info("Computing score matrix.")
    return _compute_sap(random_state, mus, ys, mus_test, ys_test, continuous_factors)


def _compute_sap(random_state, mus, ys, mus_test, ys_test, continuous_factors):
    """Computes score based on both training and testing codes and factors."""
    score_matrix, train_matrix = compute_score_matrix(random_state, mus, ys, mus_test,
                                      ys_test, continuous_factors)
    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == mus.shape[0]
    assert score_matrix.shape[1] == ys.shape[0]
    scores_dict = {}

    if continuous_factors == False:
        scores_dict["SAP_discrete"] = compute_avg_diff_top_two(score_matrix)
        scores_dict["SAP_discrete_train"] = compute_avg_diff_top_two(train_matrix)
        logging.info("SAP discrete score: %.2g", scores_dict["SAP_discrete"])
    else:
        scores_dict["SAP_continuous"] = compute_avg_diff_top_two(score_matrix)
        logging.info("SAP continuous score: %.2g", scores_dict["SAP_continuous"])

    return scores_dict




def compute_score_matrix(random_state, mus, ys, mus_test, ys_test, continuous_factors):
    """Compute score matrix as described in Section 3."""
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]

    train_score_matrix = np.zeros([num_latents, num_factors])
    result_score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):

            mu_i = mus[i, :]
            y_j = ys[j, :]

            if continuous_factors:
                # Attribute is considered continuous.
                cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1]**2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    result_score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                else:
                    result_score_matrix[i, j] = 0.
            else:
                # Attribute is considered discrete.
                mu_i = mus[i, :]
                y_j = ys[j, :]
                mu_i_test = mus_test[i, :]
                y_j_test = ys_test[j, :]
                classifier = LogisticRegression(penalty='none', max_iter=10000, random_state=random_state)
                classifier.fit(mu_i[:, np.newaxis], y_j)

                pred = classifier.predict(mu_i_test[:, np.newaxis])
                result_score_matrix[i, j] = np.mean(pred == y_j_test)

                pred = classifier.predict(mu_i[:, np.newaxis])
                train_score_matrix[i, j] = np.mean(pred == y_j)

    return result_score_matrix, train_score_matrix


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
