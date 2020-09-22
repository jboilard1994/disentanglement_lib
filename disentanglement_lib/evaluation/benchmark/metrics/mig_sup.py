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
    "mig_sup",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_mig_sup(dataholder,
                    random_state,
                    artifact_dir=None,
                    num_train=gin.REQUIRED):
    """Computes the mutual information gap.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

    Returns:
    Dict with average mutual information gap.
    """
    del artifact_dir
    logging.info("Generating training set.")
    mus_train, ys_train = utils.generate_batch_factor_code(dataholder, num_train, random_state, num_train)
    assert mus_train.shape[1] == num_train

    return _compute_mig_sup(dataholder, mus_train, ys_train)


def _compute_mig_sup(dataholder, mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}
    m = np.zeros((mus_train.shape[0], ys_train.shape[0]))
    for j, y_train in enumerate(ys_train):
        discretized_mus, bins = utils.make_discretizer(mus_train, dataholder.cumulative_dist[j])
        m[:, j] = utils.discrete_mutual_info(discretized_mus, ys_train[j].reshape((1, -1))).flatten()
        pass

    # m is [num_latents, num_factors]
    assert m.shape[0] == discretized_mus.shape[0]
    assert m.shape[1] == ys_train.shape[0]

    # Find top two factor MI for each code, get individual disentanglement scores
    entropy = utils.discrete_entropy(ys_train)
    dis = np.zeros((m.shape[0],))
    norm_dis = np.zeros((m.shape[0],))
    for i, code_MI in enumerate(m):
        idx = (-code_MI).argsort()[:2]
        j_i, j_k = idx[0], idx[1]
        dis[i] = code_MI[j_i] - code_MI[j_k]
        norm_dis[i] = code_MI[j_i]/(entropy[j_i]) - code_MI[j_k]/(entropy[j_k])
        pass

    score_dict["MIG_sup_score"] = np.mean(norm_dis)
    score_dict["MIG_sup_unnormalized"] = np.mean(dis)
    return score_dict


