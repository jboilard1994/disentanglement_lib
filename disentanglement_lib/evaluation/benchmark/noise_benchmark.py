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

from disentanglement_lib.evaluation.benchmark.scenarios import noise_dataholder
from disentanglement_lib.evaluation.benchmark.benchmark_utils import manage_processes, init_dict, add_to_dict
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics
from disentanglement_lib.evaluation.benchmark.scenarios.noise_dataholder import NoiseMode


def test_metric(config_class, num_factors, val_per_factor, index_dict, queue, noise_mode):
    # Get run parameters
    K = index_dict["K"]
    alpha = index_dict["alpha"]
    seed = index_dict["seed"]
    f = index_dict["f"]

    config_fn = config_class()

    # get params
    n_samples = noise_dataholder.NoiseDataHolder.get_expected_len(num_factors, val_per_factor, K)
    metric_fn = config_fn.get_metric_fn_id()[0]

    configs = config_fn.get_gin_configs(n_samples, val_per_factor)
    param_ids, all_params = config_fn.get_extra_params()
    results = init_dict({}, all_params, depth=0)
    
    for i, config in enumerate(configs):
        gin_config_files, gin_bindings = config
        extra_param_id = param_ids[i]
        # apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)

        # set random states & go!
        random_state = np.random.RandomState(seed)

        dataholder = noise_dataholder.NoiseDataHolder(alpha=alpha,
                                                      random_state=random_state,
                                                      K=K,
                                                      num_factors=num_factors,
                                                      val_per_factor=val_per_factor,
                                                      noise_mode=noise_mode)

        # Get scores and save in matrix
        score = metric_fn(dataholder, random_state)
        results = add_to_dict(results, extra_param_id, score, 0)
        
        gin.clear_config()
        
    return_dict = {"K": K, "alpha": alpha, "seed": seed, "f": f, "result": results}
    queue.put(return_dict)  # Multiprocessing accessible list.
    
    return return_dict


def organize_results(result_dicts_list, metric_id):
    """ Organizes input list of result dicts into indexed K, sub-index alpha, sub-sub-index (etc) depending on the metric,
    with final index being the metric name/list of seeded results"""

    # Find all unique values
    Ks = []
    alphas = []
    seeds = []
    for result_dict in result_dicts_list:
        Ks.append(result_dict["K"])
        alphas.append(result_dict["alpha"])
        seeds.append(result_dict["seed"])

    # Isolate all values
    Ks = np.unique(Ks)
    alphas = np.unique(alphas)
    seeds = np.unique(seeds)

    # initialize organized_results
    organized_results = {}
    for K in Ks:
        organized_results[K] = {}
        for alpha in alphas:
            organized_results[K][alpha] = {}

    # Fill organized dict!
    for result_dict in result_dicts_list:
        f = result_dict['f']
        K = result_dict['K']
        alpha = result_dict['alpha']
        fn_result_dict = result_dict["result"]

        # Bvae and FVAE have common extra parameters to evaluate.
        if metric_id == Metrics.BVAE or metric_id == Metrics.FVAE or metric_id == Metrics.RFVAE:
            # if a dict entry does not exist yet.
            if organized_results[K][alpha] == {}:
                for batch_size, num_eval_dict in fn_result_dict.items():
                    organized_results[K][alpha][batch_size] = {}

                    for num_eval, scores_dict in num_eval_dict.items():
                        organized_results[K][alpha][batch_size][num_eval] = {}

                        for score_name, __ in scores_dict.items():
                            organized_results[K][alpha][batch_size][num_eval][score_name] = []

            # Fill in the organized dict. append seeded results
            for batch_size, num_eval_dict in fn_result_dict.items():
                for num_eval, scores_dict in num_eval_dict.items():
                    for score_name, score in scores_dict.items():
                        organized_results[K][alpha][batch_size][num_eval][score_name].append(score)

        # All other metric organize their dictionnary here.
        else:
            # if a dict entry does not exist yet.
            if organized_results[K][alpha] == {}:
                for key, __ in fn_result_dict.items():
                    organized_results[K][alpha][key] = []

            # F ill in the organized dict. append seeded results
            for metric_name, value in fn_result_dict.items():
                organized_results[K][alpha][metric_name].append(value)

    return organized_results


def noise_scenario_main(config_fn, num_factors, val_per_factor, noise_mode, nseeds=50, process_mode="debug"):
    # define scenario parameter alpha
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]
    
    processes = []
    result_dicts_list = []
    q = mp.Queue()
     
    for K in [1, 8]:
        for alpha in alphas: # set noise strength
            
            for seed in range(nseeds):
                index_dict = {'K': K, 'alpha': alpha, 'seed': seed, 'f': str(config_fn.get_metric_fn_id()[0])}
                
                if process_mode == "debug": # allows breakpoint debug.
                    result_dicts_list.append(test_metric(config_fn, num_factors, val_per_factor, index_dict, q, noise_mode))
                    print(result_dicts_list[-1])
                    
                elif process_mode == "mp": 
                    process = mp.Process(target=test_metric,
                                         args=(config_fn,
                                               num_factors,
                                               val_per_factor,
                                               index_dict,
                                               q,
                                               noise_mode),
                                         name="Noise1_K={}, alpha={}, seed = {}, fn={}".format(K, alpha, seed, str(config_fn)))
                    
                    processes.append(process)
                
    if process_mode == "mp": 
        result_dicts_list = manage_processes(processes, q)
    
    return organize_results(result_dicts_list, config_fn.get_metric_fn_id()[1])



          
                   


  
  
  