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


class ModCompactMode(Enum):
    TEST_BLUR = 1
    TEST_CODE_FACTOR_DECAY = 2
    TEST_MOD_MISSING_CHECK = 3
    TEST_COMPACT_MISSING_CHECK = 4
    TEST_COMPACT_REDUCE = 5
    TEST_MOD_REDUCE = 6


class ModCompactDataHolder(DataHolder):
    """Author : Jonathan Boilard 2020
    Dataset where dummy factors are also the observations, with ratio of noise to relationship between code/factors."""

    def __init__(self, random_state, alpha, scenario_mode, num_factors=2, val_per_factor=10, K=1, n_extra_z=5):
        if scenario_mode == ModCompactMode.TEST_MOD_REDUCE:
            num_factors = 4

        self.alpha = alpha
        self.val_per_factor = val_per_factor
        self.mod_compact_mode = scenario_mode
        self.factor_sizes = [val_per_factor]*num_factors

        discrete_factors, continuous_factors, representations = self._load_data(random_state, num_factors, alpha)
        DataHolder.__init__(self, discrete_factors, continuous_factors, representations)
        pass
    
    @staticmethod
    def get_expected_len(num_factors, val_per_factor, K, mod_compact_mode):
        if mod_compact_mode == ModCompactMode.TEST_MOD_REDUCE:
            num_factors = 4
        return val_per_factor**num_factors

    def _load_data(self, random_state, num_factors, alpha):
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

        # Get random binned continuous factor values and get representations
        continuous_factors = []
        representations = []
        discrete_factors = []

        # Generate Continuous Features
        for discrete_features in factor_atoms:
            # Generate a continuous feature from binning possible range.
            continuous_features = []
            for i, d_feature in enumerate(discrete_features):

                continuous_vals = random_state.uniform(factor_d_bins[i][d_feature][0],  # min
                                                       factor_d_bins[i][d_feature][1])  # max

                continuous_features.append(continuous_vals)
                pass

            # generate representations which are perfect with noise induction
            continuous_factors.append(continuous_features)
            discrete_factors.append(discrete_features)

                                      # Generate representations for each set of continuous features
        R = np.identity(num_factors)  # #### code-factor relationships for each sub scenario aligned with matmuls :
                                                                        #  z0 z1 z2  #
        if self.mod_compact_mode == ModCompactMode.TEST_BLUR:           # | 1  a  a |
            R = np.ones((num_factors, num_factors))*alpha               # | a  1  a |
            for i in range(num_factors):                                # | a  a  1 |
                R[i, i] = 1

        if self.mod_compact_mode == ModCompactMode.TEST_CODE_FACTOR_DECAY:     # | 1 0  0  |
            R[num_factors - 1, num_factors - 1] = 1 - alpha                    # | 0 1  0  |
                                                                               # | 0 0 1-a |

        if self.mod_compact_mode == ModCompactMode.TEST_MOD_MISSING_CHECK:    # | 1 0 0 |
            R[num_factors-1, num_factors-2] = 1                              # | 0 1 0 |
            R[num_factors-1, num_factors-1] = 0                              # | 0 1 0 |

        if self.mod_compact_mode == ModCompactMode.TEST_COMPACT_MISSING_CHECK:  # | 1 0 0 |
            R[num_factors-2, num_factors-1] = 1                                # | 0 1 1 |
            R[num_factors-1, num_factors-1] = 0                                # | 0 0 0 |

        if self.mod_compact_mode == ModCompactMode.TEST_MOD_REDUCE:              # | 1 0 |
            R1 = np.identity(int(num_factors/2))                                     # | 0 1 |
            R2 = np.identity(int(num_factors/2))*alpha                               # | a 0 |
            R = np.concatenate((R1, R2), axis=0)                                # | 0 a | # doubles num_factors (example has 2*2 factors, 2 codes)

        if self.mod_compact_mode == ModCompactMode.TEST_COMPACT_REDUCE:          # | 1 0 0 a 0 0 |
            R1 = np.identity(num_factors)                                       # | 0 1 0 0 a 0 |
            R2 = np.identity(num_factors)*alpha                                 # | 0 0 1 0 0 a |
            R = np.concatenate((R1, R2), axis=1)

        # Generate representations
        for continuous_features in continuous_factors:
            rep = np.matmul(continuous_features, R)

            # Add noise if necessary.
            if self.mod_compact_mode == ModCompactMode.TEST_CODE_FACTOR_DECAY:
                noise = random_state.uniform(low=-1, high=1)
                rep[-1] = noise*alpha + rep[-1]

            elif self.mod_compact_mode == ModCompactMode.TEST_MOD_MISSING_CHECK:
                noise = random_state.uniform(low=-1, high=1)
                rep[-1] = noise + rep[-1]

            elif self.mod_compact_mode == ModCompactMode.TEST_COMPACT_REDUCE:
                noise = random_state.uniform(low=-1, high=1, size=(num_factors,))
                rep[num_factors:] = rep[num_factors:] + noise*(1-alpha)

            representations.append(rep)

        return np.array(discrete_factors), np.array(continuous_factors), np.array(representations)
