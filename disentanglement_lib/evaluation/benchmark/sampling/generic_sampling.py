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


class GenericSampling(object):
    """ @author: jboilard
    State space with factors split between latent variable and observations and sampling."""

    def __init__(self, factor_sizes, latent_factor_indices, factors, observation_ids):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [i for i in range(self.num_factors)
                                                    if i not in self.latent_factor_indices]
        
        self.num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = self.num_total_atoms / np.cumprod(self.factor_sizes)
        
        #atom_grouped_observations is referenced by sampling_list
        self.atom_grouped_observations, self.sampling_list, self.factor_splices_count = self.get_sampling_elements(factors, observation_ids)
        self.initial_sampling_list = copy.deepcopy(self.sampling_list)
        self.initial_factor_splices_count = copy.deepcopy(self.factor_splices_count)
        self.sampled_list = []
        self.paired_sampled_list = []
        self.first_sample = True
    
    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)
      
    def get_sampling_elements(self, factors, observation_ids):
        """ @author: jboilard
        Called at class object initialization, initializes sampling_list, 
        atom_grouped_observations and factor_splices count class objects
        returns : 
            atom_grouped_observations : orders spectrogram dataset ids according to a state-space index (atom) 
                                            and lists together ids where sapectrograms have the same factors
            sampling_list : a list to shuffle, where one sample is a pair [state-space-index, occurence], 
                                            these pairs reference the object atom_grouped_observations
            factor_splices_count : from a matrix which counts every possible factor combinations, sum the f0 dimension, f1 dimension, 
                                    etc until a sum array is obtained for every dimension. 
                                    This objhect is important to determine which factor values can or cannot be locked for the B-VAE algorithm and is updated at every sampling step
                                    splice_factor_count[n_factors, n_possible_factor_values]"""
        #Create sampling list atom_count_listand factor list
        factor_count_list = np.zeros(shape=self.factor_sizes)
        factor_splices_count = []
        
        feature_state_space_index = self._features_to_state_space_index(factors) #get state-space index of each factor
        atom_grouped_observations = []
        sampling_list = []
        for i in range(self.num_total_atoms):
            atom_grouped_observations.append([]) # initialize list
        
        #Create the atom_grouped_observation arrays and sampling_list
        for i, index in enumerate(feature_state_space_index):
            atom_grouped_observations[index].append(observation_ids[i])
            sampling_list.append([index, len(atom_grouped_observations[index]) - 1])
           
        #counts every factor combination
        for factor in factors:
            idx = tuple(factor)
            factor_count_list[idx] = factor_count_list[idx] + 1
        
        # reduce the factor combination count matrix ton 1-d by summing every other axis, for each dimension
        for i in range(self.num_factors):
            axis_squish_list = list(range(self.num_factors))
            axis_squish_list.remove(i)
            factor_splices_count.append(np.sum(factor_count_list, axis=tuple(axis_squish_list)))
            
        factor_splices_count = self.get_splices(factors, observation_ids)
            
        return atom_grouped_observations, sampling_list, factor_splices_count
    

    def get_splices(self, factors, observation_ids):
        """ @author: jboilard
        Method overrided depending on class"""                  
        return []
        
    def reset_sampling_objects(self):
        """ @author: jboilard
        Called when the sampling list is empty and more samples are required, or 
        whenever a new experiment is started. Resets sampling_list and factor_splices_count to initial values"""
        self.sampling_list = copy.deepcopy(self.initial_sampling_list)
        self.factor_splices_count = copy.deepcopy(self.initial_factor_splices_count)
    
    def remove_sample_from_lists(self, factor_id):
        """ @author: jboilard
        Called whenever an observation is sampled, remove factor/sample from sampling_list 
        also builds a "sampled list" to keep track of selected samples during evaluation"""
        #remove sample from list
        self.sampling_list.remove(factor_id)
        self.sampled_list.append(self._state_space_index_to_features(factor_id[0]))
   
    # def sample_possible_latent_factors(self, num, random_state, lock_id, possible_lock_vals):
    #     """ @author: jboilard
    #     Using a lock_id, corresponding to the locked factor, 
    #     sample another factor_set where the locked factor is the same than those specified in possible lock_vals
    #     num : num samples to sample
    #     random_state : enables repeatability of experiments by making sampling deterministic
    #     lock_id : index of locked factor (e.g. pitch is locked, corresponding to lock_id 0)
    #     possible_lock_vals : Presampled values. values in column lock_id must be the same than the "num" factors sampled """
    #     factors = []
    #     obs_id = []
        
    #     for i in range(num):
    #         factor_id = None
    #         self.sampling_list = shuffle(self.sampling_list, random_state=random_state)
            
    #         for sample in self.sampling_list:
    #             for val in possible_lock_vals:
    #                 if self._state_space_index_to_features(sample[0])[lock_id] == val:
    #                     factor_id = sample
    #                     break ## break from for possible_factor_loop
    #             if not factor_id == None:
    #                 break ## break from sample loop
                    
    #         factors.append(self._state_space_index_to_features(factor_id[0]))
    #         obs_id.append(self.atom_grouped_observations[factor_id[0]][factor_id[1]])
    #         self.remove_sample_from_lists(factor_id)
            
    #     return np.asarray(factors), obs_id
    
    def sample_latent_factors(self, num, random_state, lock_list = []):
        """ @author: jboilard
        randomly sample a batch of factors.
        num : number of factors to sample
        random_state : enables repeatability of experiments by making sampling deterministic
        lock_list : passed to function _sample_factor
        returns : factors, observation_id which indexes the Dataset Object"""
        factors = []
        obs_id = []
        for i in range(num):
            lock = [] if len(lock_list) == 0 else lock_list[i]
            factor_id = self._sample_factor(random_state, lock)
            
            # if a sample has been effectively sampled. Only goes in else if code does not work properly.
            if not factor_id == None:
                factors.append(self._state_space_index_to_features(factor_id[0]))
                obs_id.append(self.atom_grouped_observations[factor_id[0]][factor_id[1]])
                self.remove_sample_from_lists(factor_id)

            else:
                print("Warning, lock-filtered factor sample not found. Reduce num sampling, increase datasize, or ignore because of outlier data")
                factors.append([np.NaN]*len(self.factor_sizes))
                obs_id.append(None)
       
        return np.asarray(factors), obs_id
    
    
    def _sample_factor(self, random_state, lock_list = []):
        """ @author: jboilard
        function called by all other sampling methods. samples a factor corresponding to the possibilities listed in lock_list"""
        factor_id = None 
        #if no lock list, do simple random sampling
        if lock_list == []:
            sample_id = random_state.randint(len(self.sampling_list))
            factor_id = self.sampling_list[sample_id]            
        else:
            #if there is a lock list, list all possibilities, and sample a factor which corresponds to that possibility
            no_sample_available = False
            while factor_id == None and no_sample_available == False:
                possible_factors = self.get_possible_indexes(lock_list, random_state)
                self.sampling_list = shuffle(self.sampling_list, random_state=random_state)
                # possible factor list. checks sampling list in order, bu it has been pre-shuffled, making it random.
                for sample in self.sampling_list:
                    for possible_factor in possible_factors:
                        if sample[0] == possible_factor:
                            factor_id = sample

                            break ## break from for possible_factor_loop
                    if not factor_id == None:
                        break ## break from sample loop
                        
                #if sample not found, continue while loop, else break out of it.
                if factor_id == None:
                    no_sample_available = True
        return factor_id

    
    def get_possible_indexes(self, factors_lock, random_state):
        """ from defined unlocked (-1) and locked factors, get all possible values for sampling"""
        factor_list = [[]]
        for i, lock_val in enumerate(factors_lock):
            ## if unlocked, number of possible factor-samples are multiplied by the number of possible unlocked factor values
            if lock_val == -1 : 
                base = factor_list
                factor_list = []
                for b in base:
                    bc = b.copy()
                    for f in range(self.factor_sizes[i]):
                        bc.append(f)
                        factor_list.append(bc)
                        bc = b.copy()
            
            else: ## if factor is locked, just append the locked factor value
                for i in range(len(factor_list)):
                    factor_list[i].append(lock_val)
        #transform to index
        possible_indexes = self._features_to_state_space_index(np.asarray(factor_list))     
        return possible_indexes
    
    
    def sample_possible_locking(self, batchsize, random_state, mode):
        raise NotImplementedError()
        pass        


    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
    
        Args:
          features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the input space should be
            returned."""
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    
    def _features_to_state_space_index(self, features):
        """Returns the indices in the atom space for given factor configurations.
    
        Args:
          features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the atom space should be
            returned.
        """
        if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
            np.any(features < 0)):
          raise ValueError("Feature indices have to be within [0, factor_size-1]!")
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)
    
    def _state_space_index_to_features(self, index):
        """Returns the corresponding features corresponding to an id.
    
        Args:
          index: single index value representing a factor atom.
        """
        factor = []
        for base in self.factor_bases:
            f = math.floor(index/base)
            factor.append(f)
            index = index - f*base
            
        return factor
