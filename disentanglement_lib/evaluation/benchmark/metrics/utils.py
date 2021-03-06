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

"""Utility functions that are useful for the different metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
import gin.tf
from scipy import stats


def generate_batch_factor_code(dataholder,
                               num_points, random_state, batch_size, reset=False, continuous=False):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    dataholder : Contains factors and representation
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  
   
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors, current_observations_ids = \
        dataholder.sample_factors(num_points_iter, random_state)
    if i == 0:
    
      if continuous == False : 
          factors = current_factors
      else : 
          factors = np.take(dataholder.continuous_factors, current_observations_ids, axis=0)
      
      representations = np.take(dataholder.embed_codes, current_observations_ids, axis=0)
    else:
        
      if continuous == False : 
          factors = np.vstack((factors, current_factors))
      else : 
          cont_factor = np.take(dataholder.continuous_factors, current_observations_ids, axis=0)
          factors = np.vstack((factors, cont_factor))
      
      rep = np.take(dataholder.embed_codes, current_observations_ids, axis=0)
      representations = np.vstack((representations, rep))
      
    i += num_points_iter
    
  if reset == True:
      dataholder.reset()
      
  return np.transpose(representations), np.transpose(factors)


def split_train_test(observations, train_percentage):
  """Splits observations into a train and test set.

  Args:
    observations: Observations to split in train and test. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]
  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test set."
  return observations_train, observations_test


def obtain_representation(observations, representation_function, batch_size):
  """"Obtain representations from observations.

  Args:
    observations: Observations for which we compute the representation.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Batch size to compute the representation.
  Returns:
    representations: Codes (num_codes, num_points)-Numpy array.
  """
  representations = None
  num_points = observations.shape[0]
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = observations[i:i + num_points_iter]
    if i == 0:
      representations = representation_function(current_observations)
    else:
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations)


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m


def discrete_entropy(ys):
  """Compute discrete entropy using mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h

def discrete_joint_entropy(z, y):
    """ Get joint-entropy of paired one code dimension and one factor's features"""
    zy_pair = np.vstack((z,y)).T
    __, inverse = np.unique(zy_pair, axis=0,  return_inverse=True)
    
    return sklearn.metrics.mutual_info_score(inverse, inverse)

@gin.configurable(
    "discretizer", blacklist=["target"])
def make_discretizer(target, cumulative_dist=None, num_bins=gin.REQUIRED,
                     discretizer_fn=gin.REQUIRED):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins, cumulative_dist)


@gin.configurable("histogram_discretizer", blacklist=["target"])
def _histogram_discretize(target, num_bins, cumulative_dist):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target, dtype=np.int32)
    all_bins = []
    for i in range(target.shape[0]):
        counts, bins = np.histogram(target[i, :], num_bins)
        discretized[i, :] = np.digitize(target[i, :], bins[:-1]) - 1
        all_bins.append(bins)
    return discretized, all_bins


@gin.configurable("percentile_discretizer", blacklist=["target"])
def _percentile_histogram_discretize(target, num_bins, cumulative_dist):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target, dtype=np.int32)
    all_bins = []

    for i in range(target.shape[0]):
        percentiles = cumulative_dist*100

        if percentiles[0] < 0: percentiles[0] = 0
        if percentiles[-1] > 100: percentiles[-1] = 100

        bins = np.percentile(target[i], percentiles)
        discretized[i, :] = np.digitize(target[i, :], bins[:-1]) - 1
        all_bins.append(bins)
    return discretized, all_bins


def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


@gin.configurable("predictor")
def make_predictor_fn(predictor_fn=gin.REQUIRED):
  """Wrapper that creates classifiers."""
  return predictor_fn


@gin.configurable("logistic_regression_cv")
def logistic_regression_cv():
  """Logistic regression with 5 folds cross validation."""
  return LogisticRegressionCV(Cs=10, cv=KFold(n_splits=5))


@gin.configurable("gradient_boosting_classifier")
def gradient_boosting_classifier():
  """Default gradient boosting classifier."""
  return GradientBoostingClassifier()

def get_middle_bins(all_bins):
    all_mids = []
    for bins in all_bins:
        mids = np.zeros((len(bins)-1,))
        for i in range(len(bins)):
            if i < len(bins)-1:
                mids[i] = (bins[i] + bins[i+1])/2
        all_mids.append(mids)
    return all_mids
            

def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted #(n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target #(n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0] #value not array

def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))

def nmse(predicted, target):
    ''' normalized mean square error '''
    return mse(predicted, target) / np.var(target)

def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / np.std(target)

def acc(predicted, target):
    ''' Accuracy of classifier '''
    return predicted == target

