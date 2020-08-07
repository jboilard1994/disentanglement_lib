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
from disentanglement_lib.evaluation.benchmark.metrics import beta_vae, dci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import gin
import pickle

def test_metric_bvae(dataholder, random_state):   
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/beta_vae_sklearn.gin"]
    results = {}
    
    for num_train_eval in [50, 150, 300]:
              
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
    

            
def noise_experiment(test_funcs, num_factors, val_per_factor, nseeds = 50):
    
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]
    all_results = {}
    
    #Test scpecified metrics.
    for f in test_funcs:
        all_results[str(f)] = {}
        
        #Set a Dataset Size
        for K in [1, 5, 10]:
            all_results[str(f)][K] = {} #make an index for K
            
            #Set noise-signal ratio alpha
            for alpha in alphas:
                all_results[str(f)][K][alpha] = {} #make an index for alpha
                
                dataholder = scenario_noise.ScenarioNoise(alpha=alpha, 
                                                              seed=0, #Dataset Seed must be different from metric seed to evaluate metric stability
                                                              K=K,
                                                              num_factors = num_factors, 
                                                              val_per_factor = val_per_factor)
                #For each seed, get result
                for seed in range(nseeds):
                    random_state = np.random.RandomState(seed)
                    results = f(dataholder, random_state)
                    print("Alpha : {}; Seed : {}; Scores : {}".format(alpha, seed, results))
                    
                    #bvae has more parameters to evaluate.
                    if f == test_metric_bvae:
                        if all_results[str(f)][K][alpha] == {}:
                            
                            for num_eval, scores_dict in results.items():
                                all_results[str(f)][K][alpha][num_eval] = {}
                                for score_name, __ in scores_dict.items():
                                    all_results[str(f)][K][alpha][num_eval][score_name] = []
                            
                            for num_eval, scores_dict in results.items():
                                for score_name, score in scores_dict.items():
                                     all_results[str(f)][K][alpha][num_eval][score_name].append(score)
                        
                    else:
                        #if dict is empty, init dict with lists.
                        if all_results[str(f)][K][alpha] == {}:
                            for key, __ in results.items():
                                all_results[str(f)][K][alpha][key] = []
                        
                        #fill dict with function values
                        for key, value in results.items():
                            all_results[str(f)][K][alpha][key].append(value)
               
    return all_results
          
                   

def make_graphs(results_dict, num_factors, val_per_factor):
    #iterate through functions
    names = ["DCI_completeness", "DCI_disentanglement", "DCI_informativeness_test", "DCI_informativeness_train"]
    for f_name, K_dict in results_dict.items():
        
        if "test_metric_dci_L1" in f_name or "test_metric_dci_gbt" in f_name:  
            if "test_metric_dci_L1" in f_name: mode="L1"
            if "test_metric_dci_gbt" in f_name: mode="gbt"
            
            #iterate through K
            for K_val, alpha_dict in K_dict.items():
                alphas = [alpha_val for alpha_val, __ in alpha_dict.items()]
                
                complete_scores = [scores_dict["DCI_completeness"] for __, scores_dict in alpha_dict.items()]
                disentagle_scores = [scores_dict["DCI_disentanglement"] for __, scores_dict in alpha_dict.items()]
                info_test_scores = [scores_dict["DCI_informativeness_test"] for __, scores_dict in alpha_dict.items()]
                info_train_scores = [scores_dict["DCI_informativeness_train"] for __, scores_dict in alpha_dict.items()]
                
                
                scores = [complete_scores, disentagle_scores, info_test_scores, info_train_scores]
                
                for name, score in zip(names, scores):
                        
                    plt.title('{} {}: Effect of noise, K={}, {} Factors / {} values each'.format(name, mode, K_val, num_factors, val_per_factor))
                    plt.violinplot(score, showmeans=True)
                    plt.xticks(range(1,len(alphas)+1), alphas)
                    plt.xlabel("Noise-signal ratio")
                    plt.ylabel(name)
                    ylim = plt.ylim()
                    plt.ylim([0, ylim[1]])
                    plt.savefig('figs/{}_{}_K_{}.png'.format(name, mode, K_val))
                    plt.show()
            pass
        
        if "test_metric_bvae" in f_name:
            names = ["BVAE_eval_accuracy", "BVAE_train_accuracy"]
            
            #iterate through K
            for K, alpha_dict in K_dict.items():
                
                alphas = [alpha_val for alpha_val, __ in alpha_dict.items()]
                num_eval_dict_sample = alpha_dict[alphas[0]]
                num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
                
                for num_eval in num_evals:
                    eval_scores = [num_dict[num_eval][names[0]] for alpha_val, num_dict in alpha_dict.items()]
                    train_scores = [num_dict[num_eval][names[1]] for alpha_val, num_dict in alpha_dict.items()]
                    
                    scores = [eval_scores, train_scores]
                    
                    for name, score in zip(names, scores):    
                        plt.title('{}: Effect of noise, {} batches, K={}, {} Factors / {} values each'.format(name, num_eval, K, num_factors, val_per_factor))
                        plt.violinplot(eval_scores, showmeans=True)
                        plt.xticks(range(1,len(alphas)+1), alphas)
                        plt.xlabel("Noise-signal ratio")
                        plt.ylabel(name)
                        ylim = plt.ylim()
                        plt.ylim([0, ylim[1]])                   
                        plt.savefig('figs/{}_K_{}_batch_{}.png'.format(name, K, num_eval))
                        plt.show()
            pass                
         
    pass


fs = [ "test_metric_dci_L1", "test_metric_dci_gbt", "test_metric_bvae"]
test_funcs = [test_metric_dci_L1, test_metric_dci_gbt, test_metric_bvae] 

if __name__ == "__main__":
  num_factors=3
  val_per_factor=10
  n_seeds = 3
  gin.clear_config()
  
  results_dict = noise_experiment(test_funcs, num_factors=num_factors, val_per_factor=val_per_factor, nseeds = n_seeds)
  pickle.dump(results_dict, open( "results_dict.p", "wb" ))
  
  make_graphs(results_dict, num_factors, val_per_factor)
  
  
  
  