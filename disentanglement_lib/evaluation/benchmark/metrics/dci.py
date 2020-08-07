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

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.benchmark.metrics import utils
import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import gin.tf


@gin.configurable(
    "dci",
    blacklist=["dataholder", "random_state",
               "artifact_dir"])
def compute_dci(dataholder, 
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                num_eval=gin.REQUIRED,
                mode=gin.REQUIRED):
  """Computes the DCI scores according to Sec 2.

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
    Dictionary with average disentanglement score, completeness and
      informativeness (train and test).
  """
  del artifact_dir
  logging.info("Generating training set.")
  # mus_train are of shape [num_codes, num_train], while ys_train are of shape
  # [num_factors, num_train].
  mus_train, ys_train = utils.generate_batch_factor_code(dataholder, num_train,
      random_state, num_train)
  
  assert mus_train.shape[1] == num_train
  assert ys_train.shape[1] == num_train
  
  mus_test, ys_test = utils.generate_batch_factor_code(
      dataholder, num_test,
      random_state, num_test)
  
  mus_eval, ys_eval = utils.generate_batch_factor_code(
      dataholder, num_test,
      random_state, num_test)
  
  scores = _compute_dci(mus_train, ys_train, mus_test, ys_test, mus_eval, ys_eval, mode)
  return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test, mus_eval, ys_eval, mode):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  
  if mode == "gbt":
      selected_predictors = hyperparam_search_gbt(mus_train, ys_train, mus_eval, ys_eval)      
      importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, 
                                                                  ys_train,
                                                                  mus_test,
                                                                  ys_test, 
                                                                  selected_predictors)
  if mode == "L1":
      selected_predictors = hyperparam_search_LogRegL1(mus_train, ys_train, mus_eval, ys_eval)
      importance_matrix, train_err, test_err = compute_importance_LogRegL1(mus_train, 
                                                                  ys_train,
                                                                  mus_test,
                                                                  ys_test, 
                                                                  selected_predictors)
  

  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["DCI_informativeness_train"] = train_err
  scores["DCI_informativeness_test"] = test_err
  scores["DCI_disentanglement"] = disentanglement(importance_matrix)
  scores["DCI_completeness"] = completeness(importance_matrix)
  return scores


def hyperparam_search_LogRegL1(x_train, y_train, x_eval, y_eval):
    num_factors = y_train.shape[0]
    alphas = np.arange(0.005, 0.15, 0.005)
    Cs = np.abs(alphas - 1) # LogisticRegression alpha is passed as C which is inverse of regularization
    
    results = np.zeros((len(alphas), num_factors))
    models = [[None for f in range(num_factors)] for a in alphas]
    selected_predictors = [None for f in range(num_factors)]
    for a, alpha in enumerate(Cs):
        for i in range(num_factors):
            model = LogisticRegression(penalty='l1', C=alpha, solver='liblinear')
            model.fit(x_train.T, y_train[i, :])
            models[a][i] = model
            results[a, i] = np.mean(model.predict(x_eval.T) == y_eval[i, :])
    
    selected_ids = np.argmax(results, axis=0)
    for f, i in enumerate(selected_ids):
        selected_predictors[f] = models[i][f]
        
    return selected_predictors


def hyperparam_search_gbt(x_train, y_train, x_eval, y_eval, min_depth = 2, max_depth = 10):
    num_factors = y_train.shape[0]
    tree_depths = range(min_depth,max_depth+1)
    
    results = np.zeros((len(tree_depths), num_factors))
    models = [[None for f in range(num_factors)] for d in tree_depths]
    selected_predictors = [None for f in range(num_factors)]
    for d, depth in enumerate(tree_depths):
        for i in range(num_factors):
            model = GradientBoostingClassifier(n_estimators=10, max_depth = depth)
            model.fit(x_train.T, y_train[i, :])
            models[d][i] = model
            results[d, i] = np.mean(model.predict(x_eval.T) == y_eval[i, :])
    
    selected_ids = np.argmax(results, axis=0)
    for f, i in enumerate(selected_ids):
        selected_predictors[f] = models[i][f]
        
    return selected_predictors

def compute_importance_gbt(x_train, y_train, x_test, y_test, selected_predictors):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = selected_predictors[i]
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)

def compute_importance_LogRegL1(x_train, y_train, x_test, y_test, selected_predictors):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = selected_predictors[i]
    
    ### NOT SURE ABOUT THIS ONE MIGHT NOT WORK
    importance_matrix[:, i] = np.sum(np.abs(model.coef_), axis=0)
    
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)

def compute_importance_L1(x_train, y_train, x_test, y_test, selected_predictors):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = selected_predictors[i]
    importance_matrix[:, i] = np.abs(model.coef_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)
