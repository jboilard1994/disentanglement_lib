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

"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
from absl import logging
from disentanglement_lib.evaluation.benchmark.metrics import utils
import numpy as np
import gin.tf


@gin.configurable(
    "dcimig",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_dcimig(dataholder,
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED):
    """Computes the mutual information gap.

    Args:
    dataholder: Holds all factors and associated representations
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.

    Returns:
    Dict with average mutual information gap.
    """
    del artifact_dir
    logging.info("Generating training set.")
    mus_train, ys_train = utils.generate_batch_factor_code(
      dataholder, num_train,
      random_state, num_train)
    assert mus_train.shape[1] == num_train

    return _compute_dcimig(dataholder, mus_train, ys_train)


def _compute_dcimig(dataholder, mus_train, ys_train):
    """Computes score."""
    score_dict = {}
    m = np.zeros((mus_train.shape[0], ys_train.shape[0]))
    for j, y_train in enumerate(ys_train):
        discretized_mus, bins = utils.make_discretizer(mus_train, dataholder.cumulative_dist[j])
        m[:, j] = utils.discrete_mutual_info(discretized_mus, ys_train[j].reshape((1, -1))).flatten()
        pass

    assert m.shape[0] == discretized_mus.shape[0]
    assert m.shape[1] == ys_train.shape[0]

    # For score normalization
    entropy = utils.discrete_entropy(ys_train)

    # Find top two factor MI for each code, get disentanglement scores and save ids
    Dis = np.zeros((m.shape[0],))
    jis = []
    for i, code_MI in enumerate(m):
        idx = (-code_MI).argsort()[:2]
        j_i, j_k = idx[0], idx[1]
        jis.append(j_i)
        Dis[i] = code_MI[j_i] - code_MI[j_k]
        pass

    Djz = []
    # For each factor, find the code which disentangles it the most.
    for j, factor_MI in enumerate(m.T):
        II_j = []
        for i in range(m.shape[0]):
            if jis[i] == j:
              II_j.append(i)

        if len(II_j) > 0:
            max_i = np.argmax(Dis[II_j])
            k_j= II_j[max_i]
            Djz.append(Dis[k_j])
        else:
            Djz.append(0)

    score_dict["DCIMIG_unnormalized"] = np.mean(Djz)
    score_dict["DCIMIG_normalized"] = np.mean(np.divide(Djz, entropy))

    return score_dict

