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
import tensorflow_hub as hub
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth import ground_truth_data


from disentanglement_lib.evaluation.benchmark.scenarios.dataholder import DataHolder
from sklearn.utils.extmath import cartesian
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


def _load_data(dataset, postprocess_dir):
    batch_size = 32
    module_path = os.path.join(postprocess_dir, "tfhub")
    reps = []

    with hub.eval_function_for_module(module_path) as f:
        def _representation_function(x):
            """Computes representation vector for input images."""
            output = f(dict(images=x), signature="representation", as_dict=True)
            return np.array(output["default"])

        for index in range(0, len(dataset.images), batch_size):
            batch = dataset.images[index:min(index + batch_size, dataset.images.shape[0]), :]
            rep = _representation_function(batch)
            reps.append(rep)
        reps = np.vstack(reps)

    # factors
    factors = cartesian([np.array(list(range(i))) for i in dataset.factors_num_values])
    return factors, reps


def _make_input_fn(ground_truth_data, seed, num_batches=None):
    """Creates an input function for the experiments."""

    def load_dataset(batch_size):
        """TPUEstimator compatible input fuction."""
        dataset = util.tf_data_set_from_ground_truth_data(ground_truth_data, seed)
        # We need to drop the remainder as otherwise we lose the batch size in the
        # tensor shape. This has no effect as our data set is infinite.
        dataset = dataset.batch(batch_size, drop_remainder=True)
        if num_batches is not None:
            dataset = dataset.take(num_batches)
        return dataset.make_one_shot_iterator().get_next()
    return load_dataset

