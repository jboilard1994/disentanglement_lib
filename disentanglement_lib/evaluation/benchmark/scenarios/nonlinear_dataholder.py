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


# Single non-linear config
class NonlinearMode(Enum):
    SIGMOID_FAV_CONTINUOUS = 1
    SIGMOID_FAV_DISCRETE = 2
    QUADRATIC_FAV_CONTINUOUS = 3
    QUADRATIC_FAV_DISCRETE = 4


class NonlinearDataHolder(DataHolder):
    """Author : Jonathan Boilard 2020
    Dataset where dummy factors are also the observations, with ratio of noise to relationship between code/factors."""

    def __init__(self, random_state, non_linear_mode, num_factors=2, val_per_factor=10):
        self.val_per_factor = val_per_factor
        # self.noise_mode = noise_mode
        # self.n_extra_z = n_extra_z
        self.factor_sizes = [val_per_factor] * num_factors

        # parse through noise modes to define scenario configs
        if non_linear_mode == NonlinearMode.SIGMOID_FAV_CONTINUOUS or non_linear_mode == NonlinearMode.QUADRATIC_FAV_CONTINUOUS:
            self.fav_continuous = True
        elif non_linear_mode == NonlinearMode.SIGMOID_FAV_DISCRETE or non_linear_mode == NonlinearMode.QUADRATIC_FAV_DISCRETE:
            self.fav_continuous = False

        # parse through modes to define scenario configs
        if non_linear_mode == NonlinearMode.QUADRATIC_FAV_CONTINUOUS or non_linear_mode == NonlinearMode.QUADRATIC_FAV_DISCRETE:
            self.fn_class = Quadratic
        elif non_linear_mode == NonlinearMode.SIGMOID_FAV_CONTINUOUS or non_linear_mode == NonlinearMode.SIGMOID_FAV_DISCRETE:
            self.fn_class = Sigmoid

        discrete_factors, continuous_factors, representations = self._load_data(random_state, num_factors)
        DataHolder.__init__(self, discrete_factors, continuous_factors, representations)
        pass

    @staticmethod
    def get_expected_len(num_factors, val_per_factor, k):
        return k * val_per_factor ** num_factors

    def _load_data(self, random_state, num_factors):
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

        n_codes = len(self.factor_sizes)

        factor_atoms = cartesian([np.array(list(range(i))) for i in self.factor_sizes])

        # define bin discretization :
        factor_d_bins = []
        for i, n_val in enumerate(self.factor_sizes):
            discrete_bins = []
            discrete_vals = np.array(range(n_val))
            for d_val in discrete_vals:
                min_max_bins = [-1 + d_val * (2 / n_val), -1 + (d_val + 1) * (2 / n_val)]
                discrete_bins.append(min_max_bins)
            factor_d_bins.append(discrete_bins)

        # Define Diagonal perfect non-linear Relationship Matrix
        r_matrix = [[None for __ in range(num_factors)] for __ in range(n_codes)]
        for i in range(n_codes):
            f = self.fn_class(random_state)
            r_matrix[i][i] = f  # fill up diagonal.

        # Get random binned continuous factor values and get representations
        continuous_factors = []
        representations = []
        discrete_factors = []
        for discrete_features in factor_atoms:
            # Generate a continuous feature from binning possible range.
            continuous_features = []
            for i, d_feature in enumerate(discrete_features):

                if self.fav_continuous:
                    continuous_vals = random_state.uniform(factor_d_bins[i][d_feature][0],  # min
                                                                factor_d_bins[i][d_feature][1])  # max
                else:
                    continuous_vals = (factor_d_bins[i][d_feature][0] + factor_d_bins[i][d_feature][1]) / 2

                continuous_features.append(continuous_vals)

            continuous_factors.append(continuous_features)
            discrete_factors.append(discrete_features)

        # generate representation
        for continuous_features in continuous_factors:
            rep = np.zeros((n_codes,))
            for j, feature in enumerate(continuous_features):
                for i, r in enumerate(r_matrix):
                    if not r[j] is None:
                        f = r[j]
                        rep[i] = rep[i] + f(feature)
            representations.append(rep)

        """if self.extra_z:
            representations = np.asarray(representations)
            noise = random_state.normal(size=(len(representations), self.n_extra_z))
            representations = np.concatenate((representations, noise), axis=1)"""

        return np.array(discrete_factors), np.array(continuous_factors), np.array(representations)


class Sigmoid:
    """Sigmoid function :
    y = a/(e^(b(x+d))) + c
    a : Amplitude amplification
    b : X amplification / Non-linearity amplification
    d : X offset
    c : function offset
    where a = 10^r_a
    where b = 10^r_b
    - r_a and r_b allow use to sample uniformly so that odds of values between e.g. 0.5 and 1 is the same --
      than the odds of values between 1 and 2
    Ranges have been arbitrarily chosen by observing function graphing : https://www.desmos.com/calculator/ex1pfvfblq
    """

    def __init__(self, random_state):
        self.r_a = random_state.uniform(-0.3, 0.3)  # Amplitude amplification Range: [10^-0.3 - 10^0.3] --> [0.5, 2]
        self.r_b = random_state.uniform(0.5, 1)  # x-axis amplification: Range : [10^0.5 - 10^1] --> [3.16 - 10]
        self.d = random_state.uniform(-0.5, 0.5)  # X offset
        self.c = random_state.uniform(-2, 0)  # Y offset
        self.s = random_state.choice([-1, 1])  # increasing or decreasing selector (1:Decreasing, -1: Increasing)

    def __call__(self, x):
        a = 10 ** self.r_a
        b = 10 ** self.r_b
        z = self.s * b * (x + self.d)
        result = a / (np.exp(z) + 1) + self.c
        return result

class Quadratic:
    """Quadratic function :
    y = c_2*z^2 + c_1*z + c_0
    c_xx : polynomial coefficient
    z : x + x_offset
    Ranges have been arbitrarily chosen by observing function graphing : https://www.desmos.com/calculator/r8tvbt1xah
    """

    def __init__(self, random_state):
        self.c_2 = random_state.uniform(-1.5, 1.5)
        self.c_1 = random_state.uniform(-1, 1)
        self.c_0 = random_state.uniform(-0.5, 0.5)
        self.x_offset = random_state.uniform(-0.5, 0.5)

    def __call__(self, x):
        z = x + self.x_offset
        return self.c_2 * z ** 2 + self.c_1 * z + self.c_0


def function_bank():
    return [Sigmoid, Quadratic]






