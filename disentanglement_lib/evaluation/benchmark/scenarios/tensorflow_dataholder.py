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
from disentanglement_lib.data.ground_truth import named_data

import os
import tensorflow as tf

from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth import ground_truth_data


from disentanglement_lib.evaluation.benchmark.scenarios.dataholder import DataHolder

import numpy as np
from enum import Enum


class ModelMode(Enum):
    BVAE = 1


class TensorflowDataholder(DataHolder):
    """Author : Jonathan Boilard 2020
    Dataset where dummy factors are also the observations, with ratio of noise to relationship between code/factors."""

    def __init__(self, discrete_factors, representations):
        DataHolder.__init__(self, discrete_factors=discrete_factors, continuous_factors=None, representations=representations)
        pass



