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


class RotationMode(Enum):
    CONTINUOUS = 1


class RotationDataHolder(DataHolder):
    """Author : Jonathan Boilard 2020
    Dataset where dummy factors are also the observations, with ratio of noise to relationship between code/factors."""

    def __init__(self, random_state, alpha, scenario_mode, num_factors=2, val_per_factor=10, K=None, n_extra_z=5):
        self.theta = alpha
        self.val_per_factor = val_per_factor
        self.rotation_mode = scenario_mode
        self.factor_sizes = [val_per_factor]*num_factors

        discrete_factors, continuous_factors, representations = self._load_data(random_state, num_factors, alpha)
        DataHolder.__init__(self, discrete_factors, continuous_factors, representations)
        pass
    
    @staticmethod
    def get_expected_len(num_factors, val_per_factor, k, rotation_mode):
        return k*val_per_factor**num_factors

    def _load_data(self, random_state, num_factors, theta):
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
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Get random binned continuous factor values and get representations
        continuous_factors = []
        representations = []
        discrete_factors = []

        # sample discrete bins and continuous values.
        for discrete_features in factor_atoms:
            # Generate a continuous feature from binning possible range.
            continuous_features = []
            for i, d_feature in enumerate(discrete_features):

                continuous_vals = random_state.uniform(factor_d_bins[i][d_feature][0],  # min
                                                       factor_d_bins[i][d_feature][1])  # max

                continuous_features.append(continuous_vals)
                pass
            continuous_factors.append(continuous_features)
            discrete_factors.append(discrete_features)

        # Representation is a pair-wise rotation of continuous factors.
        for continuous_features in continuous_factors:
            # do rotation
            n_pairs = np.floor(num_factors/2).astype(np.int32)
            rot_rep = np.matmul(continuous_features, R)

            for i_pair in range(n_pairs):
                rot_rep[i_pair*2:i_pair*2 + 2] = np.matmul(rot_rep[i_pair*2 : i_pair*2 +2], rotation_matrix)
            representations.append(rot_rep)

        return np.array(discrete_factors), np.array(continuous_factors), np.array(representations)
