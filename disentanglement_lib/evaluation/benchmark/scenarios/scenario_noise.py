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

"""Dummy data sets used for testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util


from disentanglement_lib.evaluation.benchmark.scenarios.dataholder import DataHolder

from sklearn.utils.extmath import cartesian

import numpy as np


class ScenarioNoise(DataHolder):
  """Dataset where dummy factors are also the observations, with induced noise."""

  def __init__(self, seed, alpha, num_factors = 2, val_per_factor = 10, K=1):
      self.K = K
      self.alpha = alpha
      self.val_per_factor = val_per_factor  
      
      self.random_state = np.random.RandomState(seed)
     
      self.factor_sizes = [val_per_factor]*num_factors
      features = cartesian([np.array(list(range(i))) for i in self.factor_sizes])

      dataset_features, self.observations, representations = self._load_data(K, features, num_factors, alpha)
      
      DataHolder.__init__(self, dataset_features, representations)
      
      pass
          
  def _load_data(self, K, features, num_factors, alpha):
    #Make artificially generated code 
    dataset_features = []  
    observations = []  
    representations = []
     
    
    #Define relationship between factor features (observation) and representation
    R = np.identity(num_factors)
    
    for factor_features in features:    
        
        #An observation is normalized features between -1 and 1, basically a "ax + b" linear relationship
        observation = np.zeros((num_factors,))
        for i, __ in enumerate(factor_features):
            adjust = (self.factor_sizes[i]-1)/2
            observation[i] = factor_features[i]/adjust - 1
            
        #generate representations which are perfect with noise induction
        for k in range(K):
            noise = self.random_state.uniform(low=-1, high=1, size=num_factors)
            rep = noise*alpha + np.matmul(observation, R)*(1-alpha)
            
            dataset_features.append(factor_features)
            observations.append(observation)
            representations.append(rep)

            
    return np.array(dataset_features), np.array(observations), np.array(representations)
      
      

# if __name__ == "__main__":
#   scenario = ScenarioNoise(0)
#   random_state = np.random.RandomState(0)
#   a = scenario.sample_factors(5,random_state)
#   pass
  
  