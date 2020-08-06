import gin
import pandas as pd
import numpy as np
import os

from disentanglement_lib.evaluation.benchmark.sampling import FactorObservationSampling

class DataHolder:
    """ @author: jboilard
    For a dataset, Holds all factors and representations, saving time during representation, also allow no replacement sampling of data.""" 
    def __init__(self, factors, representations):
        
        #get factors and representation
        self.cont_factors = self.labels = None
        self.factors = factors
        self.embed_codes = representations
        
        #factor and sampling related params
        self.factors_indices = list(range(factors.shape[-1]))
        self.factor_sizes = np.add(np.max(self.factors, axis=0), 1)
        self.n_factors = len(self.factor_sizes)
        self.sampling = FactorObservationSampling(self.factor_sizes, self.factors_indices, self.factors, list(range(len(self.factors))))
        
    
    def __len__(self):
        """ @author: jboilard
        len of a class object is len of the dataset, represented by csv filepaths""" 
        return len(self.factors)
    
    @property
    def num_factors(self):
        """ @author: jboilard
        returns numbers of features used to evaluate disentanglement, features which are defined in gin .bind config file """
        return len(self.factors_list)
        
    def sample_factors(self, num, random_state, lock_id=None, factors_to_lock=None):
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
    
    
    def sample_factors_with_locking_possibilities(self, num, random_state, lock_index, possible_lock_vals):
        """ @author: jboilard 
        Used in certain metrics where samples with similar values are sampled.
        num : number of samples to get
        random_state : random seed
        lock_index : index where sampled factors must have same value
        possible_lock_vals : all possible values to sample at specified lock_index"""
        factors, observation_ids = self.sampling.sample_possible_latent_factors(num, random_state, lock_index, possible_lock_vals)
        return factors, observation_ids
        
        
            
            
            
              
          
          
          
          
          
  
          
          
          