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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression

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
  
  if mode == "RF_class" or mode == "LogRegL1":
      continuous = False
       
  if mode == "RF_reg" or mode == "Lasso":
      continuous = True
  
  # mus_train are of shape [num_codes, num_train], while ys_train are of shape
  # [num_factors, num_train].
  mus_train, ys_train = utils.generate_batch_factor_code(dataholder, num_train,
      random_state, num_train, continuous=continuous)
  
  assert mus_train.shape[1] == num_train
  assert ys_train.shape[1] == num_train

  mus_test, ys_test = utils.generate_batch_factor_code(
      dataholder, num_test,
      random_state, num_test, continuous=continuous)
  
  mus_eval, ys_eval = utils.generate_batch_factor_code(
      dataholder, num_eval,
      random_state, num_eval, continuous=continuous)
  
  scores = _compute_dci(random_state, mus_train, ys_train, mus_test, ys_test, mus_eval, ys_eval, mode)
  return scores


def _compute_dci(random_state, mus_train, ys_train, mus_test, ys_test, mus_eval, ys_eval, mode):
    """Computes score based on both training and testing codes and factors."""
    scores = {}

    # Set up models and hyperparameters
    if mode == "RF_class":  # Actually a Gradient Boosted Tree classifier
        tree_depths = range(2, 11)
        #params = [{"n_estimators": 10, "random_state": random_state} for d in tree_depths]
        params = [{"n_estimators": 10, "random_state": random_state}]
        #  models = [RandomForestClassifier for _ in range(len(params))]
        models = [GradientBoostingClassifier]
        err_fn = utils.acc
        importances_attr = 'feature_importances_'
        importances_fn = np.abs
        hyperparam_select_fn = np.argmax

    if mode == "RF_reg":
        tree_depths = range(2, 11)
        params = [{"n_estimators": 10, "max_depth": d, "random_state": random_state} for d in tree_depths]
        models = [RandomForestRegressor for _ in range(len(params))]
        err_fn = utils.nrmse
        importances_attr = 'feature_importances_'
        importances_fn = np.abs
        hyperparam_select_fn = np.argmin

    if mode == "LogRegL1":
        Cs = np.abs(np.arange(0.005, 0.15, 0.005) - 1)
        params = [{'penalty': 'l1', "C": C, "max_iter": 10e4, "solver": "liblinear", "random_state": random_state} for C in Cs]

        # Add no alpha penalty version for perfect data
        params.append({'penalty': 'none', 'max_iter': 10e5, "solver": "newton-cg", "random_state": random_state})  # -> lbfgs Creates too much not enough iterations errors.
        models = [LogisticRegression for _ in range(len(params))]

        err_fn = utils.acc
        importances_attr = 'coef_'
        importances_fn = importancefn_regl1
        hyperparam_select_fn = np.argmax

    if mode == "Lasso":
        params = [{"alpha": alpha, "max_iter": 10e5, "random_state": random_state} for alpha in np.arange(0.005, 0.15, 0.005)]
        models = [Lasso for _ in range(len(params))]

        # Add no alpha penalty version for perfect data
        params.append({})  # -> no random state for LinearRegression
        models.append(LinearRegression)

        err_fn = utils.nrmse
        importances_attr = 'coef_'
        importances_fn = np.abs
        hyperparam_select_fn = np.argmin

    selected_predictors = hyperparam_find(mus_train, ys_train, mus_eval, ys_eval, models, params, err_fn, hyperparam_select_fn)
    importance_matrix, train_err, test_err = compute_importance(mus_train,
                                                                ys_train,
                                                                mus_test,
                                                                ys_test,
                                                                selected_predictors,
                                                                err_fn,
                                                                importances_attr,
                                                                importances_fn)

    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["DCI_{}_informativeness_train".format(mode)] = train_err
    scores["DCI_{}_informativeness_test".format(mode)] = test_err
    scores["DCI_{}_disentanglement".format(mode)] = disentanglement(importance_matrix)
    scores["DCI_{}_completeness".format(mode)] = completeness(importance_matrix)
    return scores


def hyperparam_find(x_train, y_train, x_eval, y_eval, models, params, err_fn, hyperparam_select_fn):
    num_factors = y_train.shape[0]
    #Hyperparam search alpha
    results = np.zeros((len(params), num_factors))
    hyperparam_models = [[None for f in range(num_factors)] for param in params]
    
    selected_predictors = [None for f in range(num_factors)]
    for a, param in enumerate(params):
        for i in range(num_factors):

            m = models[a](**params[a])
            m.fit(x_train.T, y_train[i, :])
            hyperparam_models[a][i] = m
            results[a, i] = np.mean(err_fn(m.predict(x_eval.T), y_eval[i, :]))
            
    selected_ids = hyperparam_select_fn(results, axis=0)
    for f, i in enumerate(selected_ids):
        selected_predictors[f] = hyperparam_models[i][f]
        
    return selected_predictors

def compute_importance(x_train, y_train, x_test, y_test, selected_predictors, err_fn, importances_attr, importances_fn):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
  train_loss = []
  test_loss = []
  
  for i in range(num_factors):
    model = selected_predictors[i]
    
    r = getattr(model, importances_attr)[:, None]
    if not importances_fn == None:
        r = importances_fn(r)
        
    importance_matrix[:, i] = r.flatten()
    
    train_loss.append( err_fn(model.predict(x_train.T), y_train[i, :]) )
    test_loss.append( err_fn(model.predict(x_test.T), y_test[i, :]) )
    
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

def importancefn_regl1(r):
    r = np.abs(r)
    return np.sum(r, axis=0)

