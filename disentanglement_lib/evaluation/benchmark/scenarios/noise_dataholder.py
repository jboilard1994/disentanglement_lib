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
    NOISE_DECAY = 1
    NOISE_DECAY_EXTRA_Z = 2
    EXTRA_Z_COLLAPSED_TO_UNCOLLAPSED = 3
    IGNORE_FACTORS = 4


class NoiseDataHolder(DataHolder):
    """Author : Jonathan Boilard 2020
    Dataset where dummy factors are also the observations, with ratio of noise to relationship between code/factors."""

    def __init__(self, random_state, alpha, scenario_mode, num_factors=2, val_per_factor=10, K=1, n_extra_z=5):
        self.K = K
        self.val_per_factor = val_per_factor
        self.noise_mode = scenario_mode
        self.factor_sizes = [val_per_factor]*num_factors
        self.n_extra_z = n_extra_z  # TODO : CHANGE TO MAX_N_Z, WITH NONE VALUE IF NO EXTRA CODE
        self.alpha = alpha

        if scenario_mode == NoiseMode.EXTRA_Z_COLLAPSED_TO_UNCOLLAPSED and alpha == 0:
            alpha = 0.01
            self.alpha = 0.01
        elif scenario_mode == NoiseMode.IGNORE_FACTORS:
            self.n_extra_z = alpha

        discrete_factors, continuous_factors, representations = self._load_data(random_state, K, num_factors, alpha)
        DataHolder.__init__(self, discrete_factors, continuous_factors, representations)
        pass
    
    @staticmethod
    def get_expected_len(num_factors, val_per_factor, K, noise_mode):
        return K*val_per_factor**num_factors

    def _load_data(self, random_state, K, num_factors, alpha):
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
                continuous_vals = (factor_d_bins[i][d_feature][0] + factor_d_bins[i][d_feature][1]) / 2
                continuous_features.append(continuous_vals)
                pass

            for k in range(K):
                # generate representations which are perfect with noise induction
                if self.noise_mode == NoiseMode.NOISE_DECAY or self.noise_mode == NoiseMode.NOISE_DECAY_EXTRA_Z:
                    noise = random_state.uniform(-1, 1, size=num_factors)
                    rep = noise*alpha + np.matmul(continuous_features, R)*(1-alpha)

                # Representative codes are always perfect for this scenario
                elif self.noise_mode == NoiseMode.EXTRA_Z_COLLAPSED_TO_UNCOLLAPSED or self.noise_mode == NoiseMode.IGNORE_FACTORS:
                    rep = np.matmul(continuous_features, R)
                    if self.noise_mode == NoiseMode.IGNORE_FACTORS:
                        rep = rep[:int(num_factors-alpha)]
                        continuous_features = continuous_features[:int(num_factors-alpha)]
                        discrete_factors = discrete_factors[:int(num_factors - alpha)]

                representations.append(rep)
                continuous_factors.append(continuous_features)
                discrete_factors.append(discrete_features)

        representations = np.asarray(representations)
        # Amplitude-Fixed Noisy codes
        if not self.n_extra_z == 0 and (self.noise_mode == NoiseMode.NOISE_DECAY_EXTRA_Z or self.noise_mode == NoiseMode.IGNORE_FACTORS):
            noise = random_state.uniform(-1, 1, size=(len(representations), self.n_extra_z))
            representations = np.concatenate((representations, noise), axis=1)

        # Codes go from representing nothing, to full-strength noise which could potentially represent an unaccounted factor.
        elif self.noise_mode == NoiseMode.EXTRA_Z_COLLAPSED_TO_UNCOLLAPSED:
            noise = random_state.uniform(-1, 1, size=(len(representations), self.n_extra_z))*alpha
            representations = np.concatenate((representations, noise), axis=1)

        return np.array(discrete_factors), np.array(continuous_factors), representations
    # if __name__ == "__main__":
    #   scenario = ScenarioNoise(0)
    #   random_state = np.random.RandomState(0)
    #   a = scenario.sample_factors(5,random_state)
    #   pass
