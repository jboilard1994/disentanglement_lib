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
from enum import Enum


class NoiseMode(Enum):
    FAV_CONTINUOUS = 1
    FAV_CONTINUOUS_EXTRA_Z = 2
    FAV_CONTINUOUS_SEEDED_DATASET = 3
    FAV_DISCRETE = 4
    FAV_DISCRETE_EXTRA_Z = 5
    FAV_DISCRETE_SEEDED_DATASET = 6
    FAV_DISCRETE_ADD_NOISE = 7
    FAV_DISCRETE_ADD_NOISE_EXTRA_Z = 8


class NoiseDataHolder(DataHolder):
    """Author : Jonathan Boilard 2020
    Dataset where dummy factors are also the observations, with ratio of noise to relationship between code/factors."""

    def __init__(self, seed, alpha, noise_mode, num_factors=2, val_per_factor=10, K=1, n_extra_z=5):
        self.K = K
        self.alpha = alpha
        self.val_per_factor = val_per_factor
        self.noise_mode = noise_mode
        self.n_extra_z = n_extra_z
        self.random_state = np.random.RandomState(seed)
        self.factor_sizes = [val_per_factor]*num_factors
        """
        [NoiseMode.FAV_CONTINUOUS,
               NoiseMode.FAV_CONTINUOUS_EXTRA_Z,
               NoiseMode.FAV_CONTINUOUS_SEEDED_DATASET,
               NoiseMode.FAV_DISCRETE,
               NoiseMode.FAV_DISCRETE_EXTRA_Z,
               NoiseMode.FAV_DISCRETE_SEEDED_DATASET,
               NoiseMode.FAV_DISCRETE_ADD_NOISE,
               NoiseMode.FAV_DISCRETE_ADD_NOISE_EXTRA_Z]
        """
        # parse through noise modes to define scenario configs
        if noise_mode == NoiseMode.FAV_CONTINUOUS or noise_mode == NoiseMode.FAV_CONTINUOUS_EXTRA_Z or noise_mode == NoiseMode.FAV_CONTINUOUS_SEEDED_DATASET:
            self.fav_continuous = True
        else:
            self.fav_continuous = False

        if noise_mode == NoiseMode.FAV_CONTINUOUS_EXTRA_Z or noise_mode == NoiseMode.FAV_DISCRETE_EXTRA_Z or noise_mode == NoiseMode.FAV_DISCRETE_ADD_NOISE_EXTRA_Z:
            self.extra_z = True
        else:
            self.extra_z = False

        if noise_mode == NoiseMode.FAV_DISCRETE_SEEDED_DATASET or noise_mode == NoiseMode.FAV_CONTINUOUS_SEEDED_DATASET:
            self.seeded_dataset = True
        else:
            self.seeded_dataset = False

        if noise_mode == NoiseMode.FAV_DISCRETE_ADD_NOISE or noise_mode == NoiseMode.FAV_DISCRETE_ADD_NOISE_EXTRA_Z:
            self.add_noise = True
            self.replace_noise = False
        else:
            self.add_noise = False
            self.replace_noise = True

        discrete_factors, continuous_factors, representations = self._load_data(K, num_factors, alpha)
        DataHolder.__init__(self, discrete_factors, continuous_factors, representations)
        pass
    
    @staticmethod
    def get_expected_len(num_factors, val_per_factor, K):
        return K*val_per_factor**num_factors

    def _load_data(self, K, num_factors, alpha):
        """Author : Jonathan Boilard 2020
        Creates artificial dataset.

        Args:
        K: Number of samples for each factor atom
        num_factors : Number of generative factors
        alpha : Noise strength ratio

        Returns:
        discrete_factors : binned factors.
        continuous_factors : uniformly sampled factor between discrete bin min-max
        representation : Representation of the relationship with continuous_factors."""

        # define bin discretization :
        factor_atoms = cartesian([np.array(list(range(i))) for i in self.factor_sizes])
        factor_d_bins = []
        for i, n_val in enumerate(self.factor_sizes):

            discrete_bins = []
            discrete_vals = np.array(range(n_val))
            for d_val in discrete_vals:
                min_max_bins = [-1 + d_val*(2/n_val), -1 + (d_val+1)*(2/n_val)]
                discrete_bins.append(min_max_bins)

            factor_d_bins.append(discrete_bins)

        # Define relationship between factor features (observation) and representation
        R = np.identity(num_factors)

        # Get random binned continuous factor values and get representations
        continuous_factors = []
        representations = []
        discrete_factors = []

        for discrete_features in factor_atoms:
            # Generate a continuous feature from binning possible range.
            continuous_features = []
            for i, d_feature in enumerate(discrete_features):

                if self.fav_continuous:
                    continuous_vals = self.random_state.uniform(factor_d_bins[i][d_feature][0],  # min
                                                                factor_d_bins[i][d_feature][1])  # max
                else:
                    continuous_vals = (factor_d_bins[i][d_feature][0] + factor_d_bins[i][d_feature][1])/2

                continuous_features.append(continuous_vals)
                pass

            # generate representations which are perfect with noise induction
            for k in range(K):
                noise = self.random_state.uniform(low=-1, high=1, size=num_factors)

                if self.replace_noise:
                    rep = noise*alpha + np.matmul(continuous_features, R)*(1-alpha)
                elif self.add_noise:
                    rep = noise*alpha + np.matmul(continuous_features, R)

                representations.append(rep)
                continuous_factors.append(continuous_features)
                discrete_factors.append(discrete_features)

        if self.extra_z:
            representations = np.asarray(representations)
            noise = self.random_state.normal(size=(len(representations), self.n_extra_z))
            representations = np.concatenate((representations, noise), axis=1)

        return np.array(discrete_factors), np.array(continuous_factors), np.array(representations)
    # if __name__ == "__main__":
    #   scenario = ScenarioNoise(0)
    #   random_state = np.random.RandomState(0)
    #   a = scenario.sample_factors(5,random_state)
    #   pass

  