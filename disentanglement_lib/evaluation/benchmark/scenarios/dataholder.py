import gin
import pandas as pd
import numpy as np
import os

from disentanglement_lib.evaluation.benchmark.sampling.sampling_factor_fixed import SingleFactorFixedSampling
from disentanglement_lib.evaluation.benchmark.sampling.sampling_factor_varied import SingleFactorVariedSampling
from disentanglement_lib.evaluation.benchmark.sampling.generic_sampling import GenericSampling

@gin.configurable("dataholder")
class DataHolder:
    """ @author: jboilard
    For a dataset, Holds all factors and representations, saving time during representation, also allow no replacement sampling of data.""" 
    def __init__(self, discrete_factors, continuous_factors, representations, SamplingClass=gin.REQUIRED):
        
        #get factors and representation
        self.continuous_factors = continuous_factors
        self.factors = discrete_factors
        self.embed_codes = representations

        #factor and sampling related params
        self.factors_indices = list(range(self.factors.shape[-1]))
        self.factor_sizes = np.add(np.max(self.factors, axis=0), 1)
        self.n_factors = len(self.factor_sizes)
        
        if SamplingClass == "single_factor_fixed":
            self.sampling = SingleFactorFixedSampling(self.factor_sizes, self.factors_indices, self.factors, list(range(len(self.factors))))
        elif SamplingClass == "single_factor_varied":
            self.sampling = SingleFactorVariedSampling(self.factor_sizes, self.factors_indices, self.factors, list(range(len(self.factors))))
        else:
            self.sampling = GenericSampling(self.factor_sizes, self.factors_indices, self.factors, list(range(len(self.factors))))

        self.cumulative_dist, self.dists = _get_discrete_cumulative_distributions(discrete_factors)
        pass


    @property
    def num_factors(self):
        """ @author: jboilard
        returns numbers of features used to evaluate disentanglement, features which are defined in gin .bind config file """
        return self.n_factors
       
    @property
    def factors_num_values(self):
        return self.factor_sizes

    
    def __len__(self):
        """ @author: jboilard
        len of a class object is len of the dataset, represented by csv filepaths""" 
        return len(self.factors)
    

    def reset(self):
        self.sampling.reset_sampling_objects()
        pass
    
    def sample_factors(self, num, random_state, lock_id=None, factors_to_lock=None, reset=False):
        """ @author: jboilard 
        Feature sampling for disentanglement evaluation
        num: number of features to extract
        random_state : contains sklearn with seed object to increase experiment repeatability
        lock(optionnal) : limits possible sampled factors. elements 
            that are not -1 values limit the sampling possibility to factors 
            with the specified value at that factor position. 
            no locking if empty
        """

        # sample randomly from available bag
        factors, observation_ids = self.sampling.sample_latent_factors(num, random_state)
        
        if reset == True:
            self.reset()

        return factors, observation_ids
    
    def sample_with_locked_factors(self, random_state, lock_id, presampled_factors):
        """ @author: jboilard 
        Used in certain metrics where samples with same values are sampled.
        random_state : random seed
        lock_id : index where sampled factors muts have same value
        presampled_factors : factors sampled with "sample_factors_with_locking_possibilities()", one factor is sampled per element in this list, and must have the same value at lock_index than the sampled factor """
        lock_list = np.array(presampled_factors)
        lock_list[:,:lock_id] = -1
        lock_list[:,lock_id+1:] = -1
        
        factors, observation_ids = self.sampling.sample_latent_factors(len(presampled_factors), random_state, lock_list)
        return factors, observation_ids
    
    def sample_with_single_varied_factors(self, random_state, vary_id, presampled_factors):
        """ @author: jboilard 
        Used in certain metrics where samples with same values are sampled.
        random_state : random seed
        lock_id : index where sampled factors muts have same value
        presampled_factors : factors sampled with "sample_factors_with_locking_possibilities()", one factor is sampled per element in this list, and must have the same value at lock_index than the sampled factor """
        lock_list = np.array(presampled_factors)
        lock_list[:,vary_id] = -1
        
        factors, observation_ids = self.sampling.sample_latent_factors(len(presampled_factors), random_state, lock_list)
        return factors, observation_ids
    
    
    def sample_factors_with_locking_possibilities(self, num, random_state, lock_index, possible_lock_vals):
        """ @author: jboilard 
        Used in certain metrics where samples with similar values are sampled.
        num : number of samples to get
        random_state : random seed
        lock_index : index where sampled factors must have same value
        possible_lock_vals : all possible values to sample at specified lock_index"""
        factors, observation_ids = self.sampling.sample_possible_latent_factors(num, random_state, lock_index, possible_lock_vals)
        return factors, observation_ids
        
    def getall_representations_from_factors(self, given_factors, random_state):
        """Sample all possible observations and representations which have the specified factors"""
        given_factors = np.copy(given_factors).tolist()
        factors_ids = []
        for i, features in enumerate(self.factors.tolist()):
            if features in given_factors:
                factors_ids.append(i)
                
        return np.take(self.embed_codes, factors_ids, axis=0)       
            

def _get_discrete_cumulative_distributions(discrete_targets):
    cum_dists = []
    dists = []
    for i in range(discrete_targets.shape[1]):
        # get distributions first
        counts = np.bincount(discrete_targets[:, i])
        dist = counts/np.sum(counts)

        # then get cumulative.
        cum_dist = np.zeros_like(dist)
        for b_i in range(len(dist)):
            cum_dist[b_i] = np.sum(dist[:b_i + 1])

        cum_dist = np.insert(cum_dist, 0, 0)
        cum_dists.append(cum_dist)
        dists.append(dist)

    return cum_dists, dists
