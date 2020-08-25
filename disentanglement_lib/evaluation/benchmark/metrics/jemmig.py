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
    "jemmig",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_jemmig(dataholder,
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
  mus_train, ys_train = utils.generate_batch_factor_code(
      dataholder, num_train,
      random_state, num_train)
  assert mus_train.shape[1] == num_train
  return _compute_jemmig(mus_train, ys_train)


def _compute_jemmig(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  discretized_mus, bins = utils.make_discretizer(mus_train)
  n_bins = np.max(discretized_mus) +1
  
  m = utils.discrete_mutual_info(discretized_mus, ys_train)
  
  assert m.shape[0] == mus_train.shape[0]
  assert m.shape[1] == ys_train.shape[0]
  # m is [num_latents, num_factors]
  
  entropy_ys = utils.discrete_entropy(ys_train)
  entropy_zs = utils.discrete_entropy(discretized_mus)
  entropy_zs_max = utils.discrete_entropy(np.arange(n_bins, dtype=np.int32).reshape((1,-1)))
  sorted_m = np.sort(m, axis=0)[::-1]
  
  unnormalized_results = np.zeros((ys_train.shape[0],))
  normalized_results = np.zeros((ys_train.shape[0],))
  for i, yz_mi in enumerate(m.T):
      argmax_z = np.argmax(yz_mi)
      joint_entropy = utils.discrete_joint_entropy(discretized_mus[argmax_z,:], ys_train[i,:])
      unnormalized_results[i] = joint_entropy - sorted_m[0, i] + sorted_m[1, i]
      normalized_results[i] = (entropy_zs[argmax_z] + entropy_ys[i] - 2*sorted_m[0, i] + sorted_m[1, i])/(entropy_zs_max + entropy_ys[i]) 
      pass
  
  score_dict["JEMMIG_score"] = np.mean(unnormalized_results)
  score_dict["NORM_JEMMIG_score"] = np.mean(normalized_results)
  
  return score_dict


