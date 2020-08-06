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



def test_metric_dci(num_factors=3, val_per_factor=10):   
    
    n_seeds = 50
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/dci.gin"]
       
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]
    for mode in ["gbt", "L1"]:
        for K in [1, 5, 10]:
    
            for alpha in alphas:
                alpha = alpha
                seed_scores = np.zeros((n_seeds, 4))
                
                for seed in range(50):
                    random_state = np.random.RandomState(seed)
                    
                    #Define configs for this run
                    n_samples = val_per_factor**num_factors*K
                    gin_bindings = ["dci.num_train = {}".format(int(0.8*n_samples)),
                                    "dci.num_eval = {}".format(int(n_samples*0.1)),
                                    "dci.num_test = {}".format(int(n_samples*0.1)),
                                    "dci.mode = '{}'".format(mode)] #OR L1
    
                    #apply configs
                    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
                    
                    #Set up Scenario
                    dataholder = scenario_noise.ScenarioNoise(alpha=alpha, 
                                                              seed=0, #Dataset Seed must be different from metric seed to evaluate metric stability
                                                              K=K,
                                                              num_factors = num_factors, 
                                                              val_per_factor = val_per_factor)
    
                    
                    #Get scores and save in matrix
                    scores = dci.compute_dci(
                        dataholder, random_state)
                    
                    seed_scores[seed, :] = [scores["DCI_informativeness_train"], scores["DCI_informativeness_test"], scores["DCI_disentanglement"], scores["DCI_completeness"]]
                    print("K : {}, Alpha : {}; Seed : {}; Scores : {}".format(K, alpha, seed, scores))
                    gin.clear_config()
                
                #save scores for each alpha
                    
                DCI_informativeness_train = seed_scores[:,0]
                DCI_informativeness_test = seed_scores[:,1]
                DCI_disentanglement = seed_scores[:,2]
                DCI_completeness = seed_scores[:,3]
    
                
                gin.clear_config()
            
        
            #Create graphs
            plt.title('Effect of noise on seeded metric scores, K={}, {} Factors with {} values each'.format(K, num_factors, val_per_factor))
            plt.violinplot(DCI_disentanglement, showmeans=True)
            plt.xticks(range(1,len(alphas)+1), alphas)
            plt.xlabel("Noise-signal ratio")
            plt.ylabel("Disentanglement score")
            ylim = plt.ylim()
            plt.ylim([0, ylim[1]])
            plt.savefig('figs/DCIDisentanglement{}_K_{}.png'.format(mode, K))
            plt.show()
            
            plt.title('Effect of noise on seeded metric scores, K={}, {} Factors with {} values each'.format(K, num_factors, val_per_factor))
            plt.violinplot(DCI_completeness, showmeans=True)
            plt.xticks(range(1,len(alphas)+1), alphas)
            plt.xlabel("Noise-signal ratio")
            plt.ylabel("Completeness score")
            ylim = plt.ylim()
            plt.ylim([0, ylim[1]])
            plt.savefig('figs/DCICompleteness{}_K_{}.png'.format(mode, K))
            plt.show()
            
            plt.title('Effect of noise on seeded metric scores K={}, {} Factors with {} values each'.format(K, num_factors, val_per_factor))
            plt.violinplot(DCI_informativeness_test, showmeans=True)
            plt.xticks(range(1,len(alphas)+1), alphas)
            plt.xlabel("Noise-signal ratio")
            plt.ylabel("Informativeness score")
            ylim = plt.ylim()
            plt.ylim([0, ylim[1]])
            plt.savefig('figs/DCIInformativeness{}_K_{}.png'.format(mode, K))
            plt.show()
        
        
            
    
            


def test_metric_bvae(num_factors = 3, val_per_factor=10):   
    
    n_seeds = 50
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/beta_vae_sklearn.gin"]
    
    all_scores = []  
    #alphas = (1 - np.logspace(-2, 0, 10))
   # alphas = alphas[::-1]
    
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]
    num_factors = 3
    val_per_factor = 5
    
    for K in [1,5,10]:
        for num_train_eval in [100, 500, 2000]:
            alpha_eval_scores = []
            alpha_train_scores = []
            for alpha in alphas:
                alpha = alpha
                seed_scores = np.zeros((n_seeds, 2))
                
                for seed in range(50):
                    
                    random_state = np.random.RandomState(seed)
                    #Define configs for this run
                    gin_bindings = [
                          "beta_vae_sklearn.batch_size = 16",
                          "beta_vae_sklearn.num_train = {}".format(num_train_eval),
                          "beta_vae_sklearn.num_eval = {}".format(num_train_eval)]
                    
                    #apply configs
                    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
                    
                    #Set up Scenario
                    dataholder = scenario_noise.ScenarioNoise(alpha=alpha, 
                                                              seed=0, #Dataset Seed must be different from metric seed 
                                                              K=K,
                                                              num_factors = num_factors, 
                                                              val_per_factor = val_per_factor)
    
                    
                    #Get scores and save in matrix
                    scores = beta_vae.compute_beta_vae_sklearn(
                        dataholder, random_state)
                    seed_scores[seed, :] = [scores["BVAE_train_accuracy"], scores["BVAE_eval_accuracy"]]
                    print("num_train_eval : {}; Alpha : {}; Seed : {}; Scores : {}".format(num_train_eval, alpha, seed, scores))
                    gin.clear_config()
                
                #save scores for each alpha
                alpha_train_scores.append(seed_scores[:,0])
                alpha_eval_scores.append(seed_scores[:,1])
                
                gin.clear_config()
            
        
            #Create graphs
            plt.title('Effect of noise on seeded metric scores, {} batches, K={}, {} Factors with {} values each'.format(num_train_eval, K, num_factors, val_per_factor))
            plt.violinplot(alpha_eval_scores, showmeans=True)
            plt.xticks(range(1,len(alphas)+1), alphas)
            plt.xlabel("Noise-signal ratio")
            plt.ylabel("B-VAE score")
            ylim = plt.ylim()
            plt.ylim([0, ylim[1]])
            
            plt.savefig('figs/K_{}_batch_{}.png'.format(K, num_train_eval))
            plt.show()
            
            
        
        # #Create graphs
        # plt.title('Effect of noise on seeded metric scores, {} batches'.format(num_train_eval))
        # plt.violinplot(alpha_train_scores, showmeans=True)
        # plt.xticks(range(1,len(alphas)+1), alphas)
        # plt.xlabel("Noise-signal ratio")
        # plt.ylabel("B-VAE score")
        # ylim = plt.ylim()
        # plt.ylim([0, ylim[1]])
        # plt.show()
            
if __name__ == "__main__":
  gin.clear_config()
  test_metric_dci()