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

"""Implementation of the disentanglement metric from the FactorVAE paper.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.benchmark.metrics import utils
import numpy as np
from six.moves import range
import gin.tf


@gin.configurable(
    "factor_vae_score",
    blacklist=["dataholder", "random_state", "artifact_dir"])
def compute_factor_vae(dataholder,
                       random_state,
                       artifact_dir=None,
                       batch_size=gin.REQUIRED,
                       num_train=gin.REQUIRED,
                       num_eval=gin.REQUIRED):
    """Computes the FactorVAE disentanglement metric.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    num_variance_estimate: Number of points used to estimate global variances.

    Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
    """
    del artifact_dir
    logging.info("Computing global variances to standardise.")
    global_variances = _compute_variances(dataholder)

    active_dims = _prune_dims(global_variances)
    scores_dict = {}

    if not active_dims.any():
        scores_dict["train_accuracy"] = 0.
        scores_dict["eval_accuracy"] = 0.
        scores_dict["num_active_dims"] = 0
        return scores_dict

    logging.info("Generating training set.")
    training_votes = _generate_training_batch(dataholder,
                                              batch_size,
                                              num_train, random_state,
                                              global_variances, active_dims)

    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])

    logging.info("Evaluate training set accuracy.")
    train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
    logging.info("Training set accuracy: %.2g", train_accuracy)

    logging.info("Generating evaluation set.")
    eval_votes = _generate_training_batch(dataholder, batch_size,
                                          num_eval, random_state,
                                          global_variances, active_dims)

    logging.info("Evaluate evaluation set accuracy.")
    eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)

    logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
    scores_dict["FVAE_train_accuracy"] = train_accuracy
    scores_dict["FVAE_eval_accuracy"] = eval_accuracy
    scores_dict["num_active_dims"] = len(active_dims)
    return scores_dict


@gin.configurable("prune_dims", blacklist=["variances"])
def _prune_dims(variances, threshold=0.):
    """Mask for dimensions collapsed to the prior."""
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _compute_variances(dataholder):
    """Computes the variance for each dimension of the representation.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    Returns:
    Vector with the variance of each dimension.
    """
    representations = dataholder.embed_codes
    return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(dataholder,
                              batch_size, random_state, global_variances,
                              active_dims):
    """Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

    Returns:
    factor_index: Index of factor coordinate to be used.
    argmin: Index of representation coordinate with the least variance.
    """
    factors = []
    observationIds = []

    # Select random coordinate to keep fixed.
    index_lock, possible_lock_vals = dataholder.sampling.sample_possible_locking(batch_size, random_state, mode="fvae")

    # Sample two mini batches of latent variables with same values at locked factor.
    factor, observationId = dataholder.sample_factors_with_locking_possibilities(1, random_state, index_lock, possible_lock_vals)
    factors.append(factor[0])
    observationIds.append(observationId[0])

    for i in range(batch_size-1):
        factor2, observationId2 = dataholder.sample_with_locked_factors(random_state, index_lock, factor)
        factors.append(factor2[0])
        observationIds.append(observationId2[0])

    factors = np.asarray(factors)
    observationIds = np.asarray(observationIds)

    # Obtain the representations.
    representations = np.take(dataholder.embed_codes, observationIds, axis=0)

    #Get local variances, arg min of weighted local variance is a majority vote classifier training point
    local_variances = np.var(representations, axis=0, ddof=1)
    weighted_local_variances = local_variances/ global_variances
    for i, weighted_var in enumerate(weighted_local_variances):
        if active_dims[i] == False:
            weighted_local_variances[i] = np.finfo(float).max

    argmin = np.argmin(weighted_local_variances)
    return index_lock, argmin


def _generate_training_batch(dataholder,
                             batch_size, num_points, random_state,
                             global_variances, active_dims):
    """Sample a set of training samples based on a batch of ground-truth data.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

    Returns:
    (num_factors, dim_representation)-sized numpy array with votes.
    """
    votes = np.zeros((dataholder.num_factors, global_variances.shape[0]), dtype=np.int64)
    for _ in range(num_points):
        factor_index, argmin = _generate_training_sample(dataholder,
                                                         batch_size, random_state,
                                                         global_variances,
                                                         active_dims)
        votes[factor_index, argmin] += 1
    return votes
