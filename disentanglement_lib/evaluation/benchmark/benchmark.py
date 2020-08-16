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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from disentanglement_lib.evaluation.benchmark.metrics import beta_vae 
from disentanglement_lib.evaluation.benchmark.metrics import dci
from disentanglement_lib.evaluation.benchmark.metrics import factor_vae
from disentanglement_lib.evaluation.benchmark.metrics import mig
from disentanglement_lib.evaluation.benchmark.metrics import mig_sup
from disentanglement_lib.evaluation.benchmark.metrics import modularity_explicitness
from disentanglement_lib.evaluation.benchmark.metrics import sap_score
from disentanglement_lib.evaluation.benchmark.metrics import irs

from disentanglement_lib.evaluation.benchmark.scenarios import scenario_noise
from disentanglement_lib.evaluation.benchmark.benchmark_utils import manage_processes, init_dict, add_to_dict

def test_metric(dataholder, random_state, config_fn):
    #get params
    configs, all_params = config_fn(dataholder)
    results_dict = init_dict({}, all_params, depth=0)
    
    for gin_config_files, gin_bindings, metric_fn, add_index in configs:
        #apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
        
        #Get scores and save in matrix
        score = metric_fn(dataholder, random_state)
        results_dict = add_to_dict(results_dict, add_index, score, 0)
        
        gin.clear_config()
        dataholder.reset()
    return results_dict
    

def start_scenario(config_fn, num_factors, val_per_factor, index_dict, queue):
    #Get run parameters
    K = index_dict["K"]
    alpha = index_dict["alpha"]
    seed = index_dict["seed"]
    f = index_dict["f"]
    
    #Make dataset to run metrics on
    dataholder = scenario_noise.ScenarioNoise(alpha=alpha, 
                                              seed=0, #Dataset Seed must be different from metric seed to evaluate metric stability
                                              K=K,
                                              num_factors = num_factors, 
                                              val_per_factor = val_per_factor)      
    #set random states & go!
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    result = test_metric(dataholder, random_state, config_fn)
    
    
    return_dict = {"K":K, "alpha":alpha, "seed":seed, "f": f, "result":result}
    queue.put(return_dict) #Multiprocessing accessible list.
    
    return return_dict
 
def noise_scenario_main(config_fn, num_factors, val_per_factor, nseeds = 50, mode="debug"):
    # define scenario parameter alpha
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]
    
    processes = []
    result_dicts_list = []
    q = mp.Queue()
     
    for K in [1, 8]:
        for alpha in alphas: #set noise strength 
            
            for seed in range(nseeds):
                index_dict = {'K' : K, 'alpha' : alpha, 'seed' : seed, 'f' : str(config_fn)}
                
                if mode == "debug": #allows breakpoint debug.
                    result_dicts_list.append(start_scenario(config_fn, num_factors, val_per_factor, index_dict, q))
                    print(result_dicts_list[-1])
                    
                elif mode == "mp": #multiprocess mode.
                    process = mp.Process(target = start_scenario, 
                                      args=(config_fn, 
                                            num_factors,
                                            val_per_factor,
                                            index_dict, 
                                            q),
                                      name="Noise1_K={}, alpha={}, seed = {}, fn={}".format(K, alpha, seed, str(config_fn)))
                    
                    processes.append(process)
                
    if mode == "mp": #multiprocess mode.
        result_dicts_list = manage_processes(processes, q, max_process=4) 
    
    return organize_results(result_dicts_list)


def organize_results(result_dicts_list):
    """ Organizes input list of result dicts into indexed K, sub-index alpha, sub-sub-index (etc) depending on the metric, 
    with final index being the metric name/list of seeded results"""
     
    #Find all unique values
    Ks = []
    alphas = []
    seeds = []
    for result_dict in result_dicts_list:
        Ks.append(result_dict["K"])
        alphas.append(result_dict["alpha"])
        seeds.append(result_dict["seed"])
        
    #Isolate all values
    Ks = np.unique(Ks)
    alphas = np.unique(alphas)
    seeds = np.unique(seeds)
    
    #initialize organized_results
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
        
        #Bvae and FVAE have common extra parameters to evaluate.
        if "bvae" in f or "fvae" in f:  
            #if a dict entry does not exist yet.
            if organized_results[K][alpha] == {}:
                for batch_size, num_eval_dict in fn_result_dict.items():
                    organized_results[K][alpha][batch_size] = {}
                   
                    for num_eval, scores_dict in num_eval_dict.items():
                        organized_results[K][alpha][batch_size][num_eval] = {}
                       
                        for score_name, __ in scores_dict.items():
                            organized_results[K][alpha][batch_size][num_eval][score_name] = []
            
            #Fill in the organized dict. append seeded results
            for batch_size, num_eval_dict in fn_result_dict.items():
                for num_eval, scores_dict in num_eval_dict.items():
                    for score_name, score in scores_dict.items():
                         organized_results[K][alpha][batch_size][num_eval][score_name].append(score)
        
        
        #All other metric organize their dictionnary here.
        else:
            #if a dict entry does not exist yet.
            if organized_results[K][alpha] == {}:
                for key, __ in fn_result_dict.items():
                    organized_results[K][alpha][key] = []
           
            #Fill in the organized dict. append seeded results
            for metric_name, value in fn_result_dict.items():
                organized_results[K][alpha][metric_name].append(value)
                
    return organized_results
          
                   

def make_graphs(results_dict, num_factors, val_per_factor):
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
     
    #iterate through metrics
    for f_name, K_dict in results_dict.items(): 
        
        if "bvae" in f_name or "fvae" in f_name:
            if "bvae" in f_name: names = ["BVAE_eval_accuracy", "BVAE_train_accuracy"]
            elif "fvae" in f_name: names = ["FVAE_eval_accuracy", "FVAE_train_accuracy"]
            
            alpha_dict = list(K_dict.values())[0]
            batch_size_dict_sample = list(alpha_dict.values())[0]
            num_eval_dict_sample = list(batch_size_dict_sample.values())[0]
            
            num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
            batch_sizes = [batch_size for batch_size, __ in batch_size_dict_sample.items()]
            alphas = [alpha_val for alpha_val, __ in alpha_dict.items()]
            
            for name in names:
                for num_eval in num_evals:
                    for batch_size in batch_sizes:
                        labels = [] 
                
                        #iterate through K
                        for K, alpha_dict in K_dict.items():
                  
                            score = [batch_dict[batch_size][num_eval][name] for alpha_val, batch_dict in alpha_dict.items()]
                            add_label(plt.violinplot(score, showmeans = True), "K = {}".format(K))      
                        
                        plt.title('{}: Noise, BatchSize={}, {} batches, K={}, {} Factors / {} values each'.format(name, num_eval, batch_size, K, num_factors, val_per_factor))
                        plt.xticks(range(1,len(alphas)+1), alphas)
                        plt.xlabel("Noise-signal ratio")
                        plt.ylabel(name)
                        ylim = plt.ylim()
                        plt.ylim([0, ylim[1]])    
                        plt.legend(*zip(*labels), loc='center left', bbox_to_anchor=(1, 0.5))
                        plt.savefig('figs/{}_batch_{}.png'.format(name, num_eval), bbox_inches='tight')
                        plt.show() 
               
        if "mig" in f_name or "sap_continuous" in f_name or "irs" in f_name:     
            if "mig_sup" in f_name :  name = "MIG_sup_score"
            elif "mig" in f_name : name = "MIG_score"
            elif "sap_continuous" in f_name : name = "SAP_continuous"
            elif "irs" in f_name : name = "IRS"
            
            labels = []
            K_vals = [K_val for K_val, alpha_dict in K_dict.items()]
            
            #iterate through K
            for K_val, alpha_dict in K_dict.items():
                alphas = [alpha_val for alpha_val, __ in alpha_dict.items()]
                score = [scores_dict[name] for __, scores_dict in alpha_dict.items()]       
                add_label(plt.violinplot(score, showmeans = True), "K = {}".format(K_val))
                
            plt.title('{}: Effect of noise, {} Factors / {} values each'.format(name, num_factors, val_per_factor))
            plt.xticks(range(1,len(alphas)+1), alphas)
            plt.xlabel("Noise-signal ratio")
            plt.ylabel(name)
            ylim = plt.ylim()
            plt.ylim([0, ylim[1]])
            plt.legend(*zip(*labels), loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig('figs/{}.png'.format(name), bbox_inches='tight')
            plt.show() 
                
        
        if "modex" in f_name or "dci" in f_name or "sap_discrete" in f_name:  
            if "modex" in f_name : names = ["MODEX_modularity_score", "MODEX_modularity_oldtest_score", "MODEX_modularity_oldtrain_score", "MODEX_explicitness_score_train", "MODEX_explicitness_score_test"]
            if "dci_RF_class" in f_name: names = ["DCI_RF_class_completeness", "DCI_RF_class_disentanglement", "DCI_RF_class_informativeness_test", "DCI_RF_class_informativeness_train"]
            if "dci_RF_reg" in f_name: names = ["DCI_RF_reg_completeness", "DCI_RF_reg_disentanglement", "DCI_RF_reg_informativeness_test", "DCI_RF_reg_informativeness_train"]
            if "dci_LogregL1" in f_name: names = ["DCI_LogregL1_completeness", "DCI_LogregL1_disentanglement", "DCI_LogregL1_informativeness_test", "DCI_LogregL1_informativeness_train"]
            if "dci_Lasso" in f_name: names = ["DCI_Lasso_completeness", "DCI_Lasso_disentanglement", "DCI_Lasso_informativeness_test", "DCI_Lasso_informativeness_train"]
            if "sap_discrete" in f_name : names = ["SAP_discrete", "SAP_discrete_train"]
            
            for name in names:
                labels = []
                #iterate through K
                for K_val, alpha_dict in K_dict.items():
                    alphas = [alpha_val for alpha_val, __ in alpha_dict.items()]
                    score = [scores_dict[name] for __, scores_dict in alpha_dict.items()]
                    add_label(plt.violinplot(score), "K = {}".format(K_val))
                       
                plt.title('{}: Effect of noise, {} Factors / {} values each'.format(name, num_factors, val_per_factor))
                plt.xticks(range(1,len(alphas)+1), alphas)
                plt.xlabel("Noise-signal ratio")
                plt.ylabel(name)
                ylim = plt.ylim()
                plt.ylim([0, (1, ylim[1])[ylim[1]>1]   ])
                plt.legend(*zip(*labels), loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig('figs/{}.png'.format(name), bbox_inches='tight')
                plt.show() 
    pass


  
  
  