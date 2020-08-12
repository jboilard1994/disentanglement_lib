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
from disentanglement_lib.evaluation.benchmark.scenarios import scenario_noise
from disentanglement_lib.evaluation.benchmark.metrics import beta_vae, dci, fairness, factor_vae, mig, modularity_explicitness, sap_score, irs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import gin
import pickle
import matplotlib.patches as mpatches
import matplotlib



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

def test_sap(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/sap_score.gin"]
     
    #Define configs for this run
    n_samples = len(dataholder.embed_codes)
    n_bins = dataholder.val_per_factor
    
    gin_bindings = ["sap_score.num_train = {}".format(int(0.8*n_samples)),
                    "sap_score.num_test = {}".format(int(n_samples*0.2))]

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
    
    #Get scores and save in matrix
    scores = mig.compute_mig(dataholder, random_state)
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
    
    for num_train_eval in [50, 300, 500]:
              
        print("training fvae num_train_eval = {}".format(num_train_eval))         
        #Define configs for this run
        gin_bindings = [
              "factor_vae_score.batch_size = 16",
              "factor_vae_score.num_train = {}".format(num_train_eval),
              "factor_vae_score.num_eval = {}".format(num_train_eval)]
                
        #apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
        
        #Get scores
        scores = factor_vae.compute_factor_vae(dataholder, random_state)
          
        results[num_train_eval] = scores
        gin.clear_config()
        dataholder.reset()
        
    return results

def test_metric_bvae(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/beta_vae_sklearn.gin"]
    results = {}
    
    for num_train_eval in [50, 300, 500]:
              
        print("training bvae num_train_eval = {}".format(num_train_eval))         
        #Define configs for this run
        gin_bindings = [
              "beta_vae_sklearn.batch_size = 16",
              "beta_vae_sklearn.num_train = {}".format(num_train_eval),
              "beta_vae_sklearn.num_eval = {}".format(num_train_eval)]
                
        #apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
        
        #Get scores
        scores = beta_vae.compute_beta_vae_sklearn(dataholder, random_state)
          
        results[num_train_eval] = scores
        gin.clear_config()
        dataholder.reset()
        
    return results
        
def test_metric_dci_gbt(dataholder, random_state):
    scores = test_metric_dci(dataholder, random_state, "gbt")
    return scores
    
def test_metric_dci_L1(dataholder, random_state):
    scores = test_metric_dci(dataholder, random_state, "L1")
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
    

            
def noise_experiment(f, num_factors, val_per_factor, nseeds = 50):
    
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]
    results = {}
          
    #Set a Dataset Size
    for K in [1, 5, 10]:
        results[K] = {} #make an index for K
        
        #Set noise-signal ratio alpha
        for alpha in alphas:
            results[K][alpha] = {} #make an index for alpha
            
            dataholder = scenario_noise.ScenarioNoise(alpha=alpha, 
                                                          seed=0, #Dataset Seed must be different from metric seed to evaluate metric stability
                                                          K=K,
                                                          num_factors = num_factors, 
                                                          val_per_factor = val_per_factor)
            #For each seed, get result
            for seed in range(nseeds):
                np.random.seed(seed)
                random_state = np.random.RandomState(seed)
                seed_result = f(dataholder, random_state)
                print("Alpha : {}; Seed : {}; Scores : {}".format(alpha, seed, seed_result))
                
                #bvae has more parameters to evaluate.
                if f == test_metric_bvae or f == test_metric_fvae:
                    if results[K][alpha] == {}:
                        
                        for num_eval, scores_dict in seed_result.items():
                            results[K][alpha][num_eval] = {}
                            for score_name, __ in scores_dict.items():
                                results[K][alpha][num_eval][score_name] = []
                        
                    for num_eval, scores_dict in seed_result.items():
                        for score_name, score in scores_dict.items():
                             results[K][alpha][num_eval][score_name].append(score)
                    
                else:
                    #if dict is empty, init dict with lists.
                    if results[K][alpha] == {}:
                        for key, __ in seed_result.items():
                            results[K][alpha][key] = []
                    
                    #fill dict with function values
                    for key, value in seed_result.items():
                        results[K][alpha][key].append(value)
               
    return results
          
                   

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
            num_eval_dict_sample = list(alpha_dict.values())[0]
            num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
            alphas = [alpha_val for alpha_val, __ in alpha_dict.items()]
            
            for name in names:
                for num_eval in num_evals:
                    labels = [] 
            
                    #iterate through K
                    for K, alpha_dict in K_dict.items():
              
                        score = [num_dict[num_eval][name] for alpha_val, num_dict in alpha_dict.items()]
                        add_label(plt.violinplot(score, showmeans = True), "K = {}".format(K))      
                    
                    plt.title('{}: Effect of noise, {} batches, K={}, {} Factors / {} values each'.format(name, num_eval, K, num_factors, val_per_factor))
                    plt.xticks(range(1,len(alphas)+1), alphas)
                    plt.xlabel("Noise-signal ratio")
                    plt.ylabel(name)
                    ylim = plt.ylim()
                    plt.ylim([0, ylim[1]])    
                    plt.legend(*zip(*labels), loc=1)
                    plt.savefig('figs/{}_batch_{}.png'.format(name, num_eval))
                    plt.show() 
               
                ########## DO TEST IRS ################b 
        if "test_mig" in f_name or "test_sap" in f_name or "test_irs" in f_name:     
            if "test_mig" in f_name : name = "MIG_score"
            if "test_sap" in f_name : name = "SAP_score"
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
            plt.legend(*zip(*labels), loc=1)
            plt.savefig('figs/{}.png'.format(name))
            plt.show() 
                
        
        if "test_modex" in f_name or "test_metric_dci_L1" in f_name or "test_metric_dci_gbt" in f_name:  
            if "test_modex" in f_name : names = ["MODEX_modularity_score", "MODEX_explicitness_score_train", "MODEX_explicitness_score_test"]
            if "test_metric_dci_L1" in f_name: names = ["DCI_L1_completeness", "DCI_L1_disentanglement", "DCI_L1_informativeness_test", "DCI_L1_informativeness_train"]
            if "test_metric_dci_gbt" in f_name: names = ["DCI_gbt_completeness", "DCI_gbt_disentanglement", "DCI_gbt_informativeness_test", "DCI_gbt_informativeness_train"]
            
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
                plt.ylim([0, ylim[1]])
                plt.legend(*zip(*labels), loc=1)
                plt.savefig('figs/{}.png'.format(name))
                plt.show() 

    pass


  
  
  