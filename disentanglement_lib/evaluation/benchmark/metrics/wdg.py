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
from scipy.stats import wasserstein_distance


@gin.configurable(
    "wdg",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_wdg(dataholder,
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
    Dict with average mutual information gap."""

    del artifact_dir
    logging.info("Generating training set.")
    mus_train, ys_train = utils.generate_batch_factor_code(
      dataholder, num_train,
      random_state, num_train)
    assert mus_train.shape[1] == num_train

    return _compute_wdg(dataholder, mus_train, ys_train)


def _compute_wdg(dataholder, mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}

    wd_matrix = get_wasserstein_dependency_matrix(dataholder, mus_train, ys_train)

    sorted_wd = np.sort(wd_matrix, axis=0)[::-1]
    score_dict["WDG_score"] = np.mean(sorted_wd[0, :] - sorted_wd[1, :])

    return score_dict


def preprocess_discretizations(mus_train, cum_distribution):
    discretizations = []
    for j, j_cum_distribution in enumerate(cum_distribution):
        discretized_mus, bins = utils.make_discretizer(mus_train, j_cum_distribution)
        discretizations.append([discretized_mus, bins])
    return discretizations


def get_wasserstein_dependency_matrix(dataholder, mus_train, ys_train):
    wd_matrix = np.zeros((len(mus_train), len(ys_train)))
    discretizations = preprocess_discretizations(mus_train, dataholder.cumulative_dist)

    """
    mid_bins = utils.get_middle_bins(bins)
    z_bin_odds = []
    # Get odds q(z) of all z_i feature values in discretized_mus
    for z_bins in discretized_mus:
        z_bin_counts = np.bincount(z_bins, minlength=len(mid_bins))
        bin_odds = z_bin_counts/np.sum(z_bin_counts)
        z_bin_odds.append(bin_odds)"""

    # Get empirical odds of q(z|y) and compute W1( q(z) | q(z|y) )
    for i, code_vals in enumerate(mus_train):
        for j, factor_vals in enumerate(ys_train):

            discretized_mus, bins = discretizations[j]
            mid_bins = utils.get_middle_bins(bins)

            z_bin_odds = []
            # Get odds q(z) of all z_i feature values in discretized_mus
            for z_bins in discretized_mus:
                z_bin_counts = np.bincount(z_bins, minlength=len(mid_bins))
                bin_odds = z_bin_counts / np.sum(z_bin_counts)
                z_bin_odds.append(bin_odds)

            y_bins, counts = np.unique(factor_vals, return_counts=True)
            y_odds = counts/np.sum(counts)

            # get wasserstein dependency
            wd = 0
            for y_bin, y_odd in zip(y_bins, y_odds):
                inds = np.argwhere(factor_vals == y_bin)
                eval_z = discretized_mus[i][inds].flatten()
                 
                counts = np.bincount(eval_z, minlength=len(mid_bins[i]))  # force n_bins for good unique count
                z_cond_y_bin_odds = counts/np.sum(counts)

                w1 = wasserstein_distance(mid_bins[i], mid_bins[i], z_cond_y_bin_odds, z_bin_odds[i])

                wd = wd + y_odd*w1
                pass
            
            wd_matrix[i, j] = wd
          
    return wd_matrix
