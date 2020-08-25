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

"""Various utilities used in the data set code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
from six.moves import range
from sklearn.utils import shuffle
import copy

from disentanglement_lib.evaluation.benchmark.sampling.generic_sampling import GenericSampling


class SingleFactorVariedSampling(GenericSampling):
    """ @author: jboilard
    State space with factors split between latent variable and observations and sampling.
    Sampling methods adapted to favrying a factor and fixing all others"""

    def __init__(self, factor_sizes, latent_factor_indices, factors, observation_ids):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        
        #init generic class.
        GenericSampling.__init__(self, factor_sizes, latent_factor_indices, factors, observation_ids)
       
    
      
    def get_splices(self, factors, observation_ids):
        """ @author: jboilard
        Called at class object initialization, initializes sampling_list, factor_splices count objects
        returns : 
            factor_splices_count : from a matrix which counts every possible factor combinations, sum the f0 dimension, f1 dimension, 
                                    etc until a sum array is obtained for every dimension. 
                                    This objhect is important to determine which factor values can or cannot be locked for the B-VAE algorithm and is updated at every sampling step
                                    splice_factor_count[n_factors, n_possible_factor_values]"""
                                    
        factor_count_list = np.zeros(shape=self.factor_sizes)
        #counts every factor combination
        for factor in factors:
            idx = tuple(factor)
            factor_count_list[idx] = factor_count_list[idx] + 1
        
        factor_splices_arrays = []
        #define bin discretization:
        
        # reduce the factor combination count matrix ton 1-d by summing every other axis, for each dimension
        for i in range(self.num_factors):
            factor_vary_splice = np.sum(factor_count_list, axis=i)
            factor_splices_arrays.append(factor_vary_splice)

            
        return factor_splices_arrays
    
    def sample_possible_factor_to_vary(self, batchsize, random_state):
        """ @author: jboilard
        Using the factor_splices_count matrix, determine which factors can be locked, and at which values
        batch_size : the sample size for the algorithm. 
        random_state : enables repeatability of experiments by making sampling deterministic"""
        possible_lock_vals = []
        all_masked = []
        sum_masks = np.zeros((self.num_factors,))

        try: 
            # apply a binary mask to factor_splices_count, and check if elements are > treshold, checks for possible to lock factor values
            for i, splice in enumerate(self.factor_splices_count):
                masked_splice = np.where(splice > batchsize, 1, 0)
                all_masked.append(masked_splice)
                sum_masks[i] = np.sum(masked_splice)
            
            #Check if any one factor has not enough combinations of other factors to be locked
            if self.first_sample == True and not np.all(sum_masks):
                raise ValueError("HYPERPARAMETER ERROR: At least one factor has not enough combinations of other factors to be locked")
            self.first_sample = False
            
            # if any factor is still lockable, build a probability distribution to 
            # uniformly select a factor to lock at available values
            if np.sum(sum_masks) > 0:
                probs = np.where(sum_masks > 0, 1, 0)
                probs = probs/sum(probs)
                vary_index = random_state.choice(list(range(self.num_factors)), p=probs)
                
                possible_lock_vals =  np.vstack(np.where(all_masked[vary_index] == 1)).T
                    
            else:
                #if no sample locking is available, its time to reset sampling_elements!
                self.reset_sampling_objects()
                vary_index, possible_lock_vals = self.sample_possible_factor_to_vary(batchsize, random_state)
            
        except ValueError as err:
            print("HYPERPARAMETER ERROR: At least one factor has not enough combinations of other factors to be locked")
            raise err
    
        return vary_index, possible_lock_vals
    
    def sample_possible_latent_factors(self, num, random_state, index, possible_lock_vals):
        """ @author: jboilard
        Using an index corresponding to the varied factor, 
        sample a factor_set where the other factors are the same than those specified in possible lock_vals
        num : num samples to sample
        random_state : enables repeatability of experiments by making sampling deterministic
        index : index of varied factor (e.g. pitch is locked, corresponding to lock_id 0)
        possible_lock_vals : Presampled values. possible values of other factors. """
        factors = []
        obs_id = []
        index_to_lock = tuple(list(range(index)) + list(range(index+1, self.num_factors)))
        
        for i in range(num):
            factor_id = None
            self.sampling_list = shuffle(self.sampling_list, random_state=random_state)
            
            for sample in self.sampling_list:
                for val in possible_lock_vals:
                    
                    sample_features = self._state_space_index_to_features(sample[0])
                    sample_features_to_lock = sample_features[:index] + sample_features[index+1:]
                    if sample_features_to_lock == val.tolist():
                        factor_id = sample
                        break ## break from for possible_factor_loop
                if not factor_id == None:
                    break ## break from sample loop
                    
            factors.append(self._state_space_index_to_features(factor_id[0]))
            obs_id.append(self.atom_grouped_observations[factor_id[0]][factor_id[1]])
            self.remove_sample_from_lists(factor_id)
            
        return np.asarray(factors), obs_id
    
    
    def remove_sample_from_lists(self, factor_id):
        """ @author: jboilard
        Called whenever an observation is sampled, remove factor/sample from sampling_list 
        and update factors_splices_count to avoid locking factors which will not result in 
        a proper B-VAE mean code representation due to unavailable factor-pair during lock-sampling.
        also builds a "sampled list" to keep track of selected samples during evaluation"""
        #remove sample from list
        self.sampling_list.remove(factor_id)
        self.sampled_list.append(self._state_space_index_to_features(factor_id[0]))
        
        #substract 1 to factor_splices_count at indexes corresponding to sampled factors
        features = self._state_space_index_to_features(factor_id[0])
        for i, __ in enumerate(features):
            other_features = features[:i] + features[i+1:]
            val = self.factor_splices_count[i][tuple(other_features)]
            self.factor_splices_count[i][tuple(other_features)] = val-1
            
        
    
    
