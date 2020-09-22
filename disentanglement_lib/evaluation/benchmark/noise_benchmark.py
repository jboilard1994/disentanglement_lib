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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import multiprocessing as mp
import gin


from disentanglement_lib.evaluation.benchmark.metrics import beta_vae 
from disentanglement_lib.evaluation.benchmark.metrics import dci
from disentanglement_lib.evaluation.benchmark.metrics import factor_vae
from disentanglement_lib.evaluation.benchmark.metrics import mig
from disentanglement_lib.evaluation.benchmark.metrics import mig_sup
from disentanglement_lib.evaluation.benchmark.metrics import modularity_explicitness
from disentanglement_lib.evaluation.benchmark.metrics import sap_score
from disentanglement_lib.evaluation.benchmark.metrics import irs
from disentanglement_lib.evaluation.benchmark.metrics import wdg

from disentanglement_lib.evaluation.benchmark.sampling.sampling_factor_fixed import SingleFactorFixedSampling
from disentanglement_lib.evaluation.benchmark.sampling.sampling_factor_varied import SingleFactorVariedSampling
from disentanglement_lib.evaluation.benchmark.sampling.generic_sampling import GenericSampling

from disentanglement_lib.evaluation.benchmark.benchmark_utils import manage_processes, init_dict, add_to_dict, organize_results
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics


def test_metric(dataholder_class, config_class, num_factors, val_per_factor, index_dict, queue, scenario_mode):
    # Get run parameters
    K = index_dict["K"]
    alpha = index_dict["alpha"]
    seed = index_dict["seed"]
    f = index_dict["f"]

    config_ = config_class()

    # get params
    n_samples = dataholder_class.get_expected_len(num_factors, val_per_factor, K, scenario_mode)
    metric_fn = config_.get_metric_fn_id()[0]

    configs = config_.get_gin_configs(n_samples, val_per_factor)
    param_ids, all_params, param_names = config_.get_extra_params()

    results = []
    for i, config in enumerate(configs):
        gin_config_files, gin_bindings = config
        extra_param_id = param_ids[i]
        # apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)

        # set random states & go!
        random_state = np.random.RandomState(seed)

        dataholder = dataholder_class(alpha=alpha,
                                      random_state=random_state,
                                      K=K,
                                      num_factors=num_factors,
                                      val_per_factor=val_per_factor,
                                      scenario_mode=scenario_mode)

        # Get scores and save in matrix
        score = metric_fn(dataholder, random_state)
        result_dict = {"K": K, "alpha": alpha, "seed": seed, "f": f, "score": score, "extra_params": extra_param_id, "param_names": param_names}
        results.append(result_dict)
        gin.clear_config()

    queue.put(results)  # Multiprocessing accessible list.
    return results


def benchmark_main(dataholder_class, config_fn, num_factors, val_per_factor, scenario_mode, nseeds, alphas=[0], ks=[1], process_mode="mp"):
    # define scenario parameter alpha
    processes = []
    result_dicts_list = []
    q = mp.Queue()
     
    for K in ks:
        for alpha in alphas:
            
            for seed in range(nseeds):
                index_dict = {'K': K, 'alpha': alpha, 'seed': seed, 'f': str(config_fn.get_metric_fn_id()[0])}
                
                if process_mode == "debug": # allows breakpoint debug.
                    result_dicts_list.append(test_metric(dataholder_class, config_fn, num_factors, val_per_factor, index_dict, q, scenario_mode))
                    print(result_dicts_list[-1])
                    
                elif process_mode == "mp": 
                    process = mp.Process(target=test_metric,
                                         args=(dataholder_class,
                                               config_fn,
                                               num_factors,
                                               val_per_factor,
                                               index_dict,
                                               q,
                                               scenario_mode),
                                         name="{}_K={}, alpha={}, seed = {}, fn={}".format(str(scenario_mode), K, alpha, seed, str(config_fn)))
                    
                    processes.append(process)
                
    if process_mode == "mp": 
        result_dicts_list = manage_processes(processes, q)
    
    return organize_results(result_dicts_list, config_fn)
