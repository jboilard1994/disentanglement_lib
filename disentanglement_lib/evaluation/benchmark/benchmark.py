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

"""Tests for beta_vae.py."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import multiprocessing as mp
import gin
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib

from disentanglement_lib.evaluation.benchmark.scenarios import scenario_noise
from disentanglement_lib.evaluation.benchmark.metrics import beta_vae, dci, fairness, factor_vae, mig, modularity_explicitness, sap_score, irs

from disentanglement_lib.evaluation.benchmark.benchmark_utils import manage_processes

def test_irs(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/irs.gin"]
     
    #Define configs for this run
    n_samples = len(dataholder.embed_codes)
    n_bins = dataholder.val_per_factor
    
    gin_bindings = ["irs.num_train = {}".format(n_samples),
                    "irs.batch_size = 16",
                    "discretizer.discretizer_fn = @histogram_discretizer", 
                    "discretizer.num_bins = {}".format(n_bins)] #OR L1

    #apply configs
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    
    #Get scores and save in matrix
    scores = irs.compute_irs(dataholder, random_state)
    gin.clear_config()
    dataholder.reset()
    
    return scores

def test_sap_discrete(dataholder, random_state):
    return test_sap(dataholder, random_state, continuous=False)
    
def test_sap_continuous(dataholder, random_state):
    return test_sap(dataholder, random_state, continuous=True)

def test_sap(dataholder, random_state, continuous):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/sap_score.gin"]
     
    #Define configs for this run
    n_samples = len(dataholder.embed_codes)
    n_bins = dataholder.val_per_factor
    
    gin_bindings = ["sap_score.num_train = {}".format(int(0.8*n_samples)),
                    "sap_score.num_test = {}".format(int(n_samples*0.2)),
                    "sap_score.continuous_factors = {}".format(continuous)]
    
    #apply configs
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    
    #Get scores and save in matrix
    scores = sap_score.compute_sap(dataholder, random_state)
    gin.clear_config()
    dataholder.reset()
    
    return scores

def test_modex(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/modularity_explicitness.gin"]
     
    #Define configs for this run
    n_samples = len(dataholder.embed_codes)
    n_bins = dataholder.val_per_factor
    
    gin_bindings = ["modularity_explicitness.num_train = {}".format(int(0.8*n_samples)),
                    "modularity_explicitness.num_test = {}".format(int(n_samples*0.2)),
                    "discretizer.discretizer_fn = @histogram_discretizer", 
                    "discretizer.num_bins = {}".format(n_bins)] #OR L1

    #apply configs
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    
    #Get scores and save in matrix
    scores = modularity_explicitness.compute_modularity_explicitness(dataholder, random_state)
    gin.clear_config()
    dataholder.reset()
    
    return scores

def test_fairness(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/fairness.gin"]

    #Define configs for this run
    gin_bindings = [
          "fairness.num_train = 1000",
          "fairness.num_test_points_per_class=50",
          "predictor.predictor_fn = @gradient_boosting_classifier"]
            
    #apply configs
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    
    #Get scores
    scores = fairness.compute_fairness(dataholder, random_state)
     
    gin.clear_config()
    dataholder.reset()
        
    return scores


def test_metric_fvae(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/factor_vae_metric.gin"]
    
    results = {}
    batch_size=16
    results[batch_size] = {}
    
    for num_train_eval in [50, 300, 500]:
              
        print("training fvae num_train_eval = {}".format(num_train_eval))         
        #Define configs for this run
        gin_bindings = [
              "factor_vae_score.batch_size = {}".format(batch_size),
              "factor_vae_score.num_train = {}".format(num_train_eval),
              "factor_vae_score.num_eval = {}".format(num_train_eval)]
                
        #apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
        
        #Get scores
        scores = factor_vae.compute_factor_vae(dataholder, random_state)
          
        results[batch_size][num_train_eval] = scores
        gin.clear_config()
        dataholder.reset()
        
    return results

def test_metric_bvae(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/beta_vae_sklearn.gin"]
    
    results = {}
    batch_size=16
    results[batch_size] = {}
    
    for num_train_eval in [50, 300, 500]:
              
        print("training bvae num_train_eval = {}".format(num_train_eval))         
        #Define configs for this run
        gin_bindings = [
              "beta_vae_sklearn.batch_size = {}".format(batch_size),
              "beta_vae_sklearn.num_train = {}".format(num_train_eval),
              "beta_vae_sklearn.num_eval = {}".format(num_train_eval)]
                
        #apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
        
        #Get scores
        scores = beta_vae.compute_beta_vae_sklearn(dataholder, random_state)
          
        results[batch_size][num_train_eval] = scores
        
        gin.clear_config()
        dataholder.reset()
        
    return results
        
def test_metric_dci_RF_class(dataholder, random_state):
    scores = test_metric_dci(dataholder, random_state, "RF_class")
    return scores
    
def test_metric_dci_RF_reg(dataholder, random_state):
    scores = test_metric_dci(dataholder, random_state, "RF_reg")
    return scores

def test_metric_dci_LogregL1(dataholder, random_state):
    scores = test_metric_dci(dataholder, random_state, "LogregL1")
    return scores
    
def test_metric_dci_Lasso(dataholder, random_state):
    scores = test_metric_dci(dataholder, random_state, "Lasso")
    return scores

def test_metric_dci(dataholder, random_state, mode):   
    
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/dci.gin"]
    #Define configs for this run
    n_samples = len(dataholder.embed_codes)
    gin_bindings = ["dci.num_train = {}".format(int(0.8*n_samples)),
                    "dci.num_eval = {}".format(int(n_samples*0.1)),
                    "dci.num_test = {}".format(int(n_samples*0.1)),
                    "dci.mode = '{}'".format(mode)] #OR L1

    #apply configs
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    
    #Get scores and save in matrix
    scores = dci.compute_dci(dataholder, random_state)
    gin.clear_config()
    dataholder.reset()
    return scores

def test_mig(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/mig.gin"]
     
    #Define configs for this run
    n_samples = len(dataholder.embed_codes)
    n_bins = dataholder.val_per_factor
    
    
    gin_bindings = ["mig.num_train = {}".format(n_samples),
                    "discretizer.discretizer_fn = @histogram_discretizer",
                    "discretizer.num_bins = {}".format(n_bins)]

    #apply configs
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    
    # #Get scores and save in matrix
    scores = mig.compute_mig(dataholder, random_state)
    gin.clear_config()
    dataholder.reset()
    
    return scores

def mp_fn(fn, num_factors, val_per_factor, index_dict, queue):
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
    # #set random states
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    
    #Go!
    result = fn(dataholder, random_state)
    queue.put({"K":K, "alpha":alpha, "seed":seed, "f": f, "result":result})
    pass
 
def noise_experiment(f, num_factors, val_per_factor, nseeds = 50):
    # define scenario parameter alpha
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]
    
    processes = []
    q = mp.Queue()
          
    #Set a Dataset Size
    for K in [1, 8]:
        for alpha in alphas:  
            #For each seed, get result
            for seed in range(nseeds):
                process = mp.Process(target = mp_fn, 
                                     args=(f, num_factors,
                                           val_per_factor,
                                           {'K' : K, 'alpha' : alpha, 'seed' : seed, 'f' : str(f)}, 
                                           q),
                                     name="Noise1_K={}, alpha={}, seed = {}, fn={}".format(K, alpha, seed, str(f)))
                
                processes.append(process)

    result_dicts_list = manage_processes(processes, q, max_process=4) 
    results = organize_results(result_dicts_list)
    return results


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
        if "test_metric_bvae" in f or "test_metric_fvae" in f:  
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
        
        if "test_metric_bvae" in f_name or "test_metric_fvae" in f_name:
            if "test_metric_bvae" in f_name: names = ["BVAE_eval_accuracy", "BVAE_train_accuracy"]
            if "test_metric_fvae" in f_name: names = ["FVAE_eval_accuracy", "FVAE_train_accuracy"]
            
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
                        plt.savefig('figs/{}_batch_{}.png'.format(name, num_eval))
                        plt.show() 
               
        if "test_mig" in f_name or "test_sap_continuous" in f_name or "test_irs" in f_name:     
            if "test_mig" in f_name : name = "MIG_score"
            if "test_sap_continuous" in f_name : name = "SAP_continuous"
            if "test_irs" in f_name : name = "IRS"
            
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
            plt.savefig('figs/{}.png'.format(name))
            plt.show() 
                
        
        if "test_modex" in f_name or "test_metric_dci" in f_name or "test_sap_discrete" in f_name:  
            if "test_modex" in f_name : names = ["MODEX_modularity_score", "MODEX_explicitness_score_train", "MODEX_explicitness_score_test"]
            if "test_metric_dci_RF_class" in f_name: names = ["DCI_RF_class_completeness", "DCI_RF_class_disentanglement", "DCI_RF_class_informativeness_test", "DCI_RF_class_informativeness_train"]
            if "test_metric_dci_RF_reg" in f_name: names = ["DCI_RF_reg_completeness", "DCI_RF_reg_disentanglement", "DCI_RF_reg_informativeness_test", "DCI_RF_reg_informativeness_train"]
            if "test_metric_dci_LogregL1" in f_name: names = ["DCI_LogregL1_completeness", "DCI_LogregL1_disentanglement", "DCI_LogregL1_informativeness_test", "DCI_LogregL1_informativeness_train"]
            if "test_metric_dci_Lasso" in f_name: names = ["DCI_Lasso_completeness", "DCI_Lasso_disentanglement", "DCI_Lasso_informativeness_test", "DCI_Lasso_informativeness_train"]
            if "test_sap_discrete" in f_name : names = ["SAP_discrete", "SAP_discrete_train"]
            
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
                plt.savefig('figs/{}.png'.format(name))
                plt.show() 

    pass


  
  
  