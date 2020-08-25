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
    Dict with average mutual information gap.
  """
  del artifact_dir
  logging.info("Generating training set.")
  mus_train, ys_train = utils.generate_batch_factor_code(
      dataholder, num_train,
      random_state, num_train)
  assert mus_train.shape[1] == num_train
  return _compute_wdg(mus_train, ys_train)


def _compute_wdg(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  discretized_mus, bins = utils.make_discretizer(mus_train)
  mid_bins = utils.get_middle_bins(bins)
  WD_matrix = get_wasserstein_dependency_matrix(discretized_mus, mid_bins, ys_train)
  
  sorted_WD = np.sort(WD_matrix, axis=0)[::-1]
  score_dict["WDG_score"] = np.mean(sorted_WD[0, :] - sorted_WD[1, :])
  
  return score_dict      
          
          
def get_wasserstein_dependency_matrix(discretized_mus, mid_bins, ys_train):
    Iw_matrix = np.zeros((len(discretized_mus), len(ys_train)))
    
    z_odds = []
    #Get odds q(z) of all z_i feature values in discretized_mus
    for z_vals in discretized_mus:
        uniques, counts = np.unique(z_vals, return_counts=True)
        odds = counts/np.sum(counts)
        z_odds.append(odds)
    
    
    #Get empirical odds of q(z|y) and compute W1( q(z) | q(z|y) )
    for i, code_vals in enumerate(discretized_mus):
        for j, factor_vals in enumerate(ys_train):
            y_bins, counts = np.unique(factor_vals, return_counts=True)
            y_odds = counts/np.sum(counts)
            
            Iw = 0
            k = 0
            for y_bin, odd in zip(uniques, odds):
                inds = np.argwhere(factor_vals == y_bin)
                eval_z = code_vals[inds]
                 
                counts, __ = np.histogram(eval_z, 
                             bins = np.max(code_vals) + 1, 
                             range=(np.min(code_vals)-0.5, np.max(code_vals)+0.5)) #force n_bins for good unique count
                 
                z_y_odds = counts/np.sum(counts)  
                W1 = wasserstein_distance(mid_bins, mid_bins, z_y_odds, z_odds[i])

                
                Iw = Iw + y_odds[k]*W1
                k=k+1
                pass
            
            Iw_matrix[i,j] = Iw
          
    return Iw_matrix


      
















  
  # score_dict["MIG_score"] = np.mean(
  #     np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
  # score_dict["MIG_unnormalized"] = np.mean(sorted_m[0, :] - sorted_m[1, :])
  # return score_dict


