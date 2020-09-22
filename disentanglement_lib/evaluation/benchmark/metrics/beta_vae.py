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

"""Implementation of the disentanglement metric from the BetaVAE paper.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework" (https://openreview.net/forum?id=Sy2fzU9gl).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
from six.moves import range
from sklearn import linear_model
import gin.tf


@gin.configurable(
    "beta_vae_sklearn",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_beta_vae_sklearn(dataholder,
                             random_state,
                             artifact_dir=None,
                             batch_size=gin.REQUIRED,
                             num_train=gin.REQUIRED,
                             num_eval=gin.REQUIRED):
  """Computes the BetaVAE disentanglement metric using scikit-learn.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
  del artifact_dir
  logging.info("Generating training set.")
  train_points, train_labels = _generate_training_batch(
      dataholder, batch_size, num_train,
      random_state)

  logging.info("Training sklearn model.")
  model = linear_model.LogisticRegression(random_state=random_state)
  model.fit(train_points, train_labels)

  logging.info("Evaluate training set accuracy.")
  train_accuracy = model.score(train_points, train_labels)
  train_accuracy = np.mean(model.predict(train_points) == train_labels)
  logging.info("Training set accuracy: %.2g", train_accuracy)

  logging.info("Generating evaluation set.")
  eval_points, eval_labels = _generate_training_batch(
      dataholder, batch_size, num_eval,
      random_state)

  logging.info("Evaluate evaluation set accuracy.")
  eval_accuracy = model.score(eval_points, eval_labels)
  logging.info("Evaluation set accuracy: %.2g", eval_accuracy)


  scores_dict = {}
  n_factors = dataholder.factors.shape[1]
  scores_dict["BVAE_train_accuracy"] = (train_accuracy - 1/n_factors)/((n_factors - 1) / n_factors)
  scores_dict["BVAE_eval_accuracy"] = (eval_accuracy - 1/n_factors)/((n_factors - 1) / n_factors)
  return scores_dict


def _generate_training_batch(dataholder,
                             batch_size, num_points, random_state):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.

  Returns:
    points: (num_points, dim_representation)-sized numpy array with training set
      features.
    labels: (num_points)-sized numpy array with training set labels.
  """
  points = None  # Dimensionality depends on the representation function.
  labels = np.zeros(num_points, dtype=np.int64)
  for i in range(num_points):
    labels[i], feature_vector = _generate_training_sample(
        dataholder, batch_size, random_state)
    if points is None:
      points = np.zeros((num_points, feature_vector.shape[0]))
    points[i, :] = feature_vector
  return points, labels


def _generate_training_sample(dataholder,
                              batch_size, random_state):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample
    random_state: Numpy random state used for randomness.

  Returns:
    index: Index of coordinate to be used.
    feature_vector: Feature vector of training sample.
  """
  # Select random coordinate to keep fixed.
  index_lock, possible_lock_vals = dataholder.sampling.sample_possible_locking(batch_size, random_state, mode="bvae")
  
  # Sample two mini batches of latent variables.
  factors1, observationIds1 = dataholder.sample_factors_with_locking_possibilities(batch_size, random_state, index_lock, possible_lock_vals)
  factors2, observationIds2 = dataholder.sample_with_locked_factors(random_state, index_lock, factors1)
  
  # Take representations associated with observations.
  representation1 = np.take(dataholder.embed_codes, observationIds1, axis=0)
  representation2 = np.take(dataholder.embed_codes, observationIds2, axis=0)
  
  # Compute the feature vector based on differences in representation.
  feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
  
  return index_lock, feature_vector