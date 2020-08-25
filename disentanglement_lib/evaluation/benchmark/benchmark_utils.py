# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:28:11 2020

@author: Jonathan Boilard
"""

import copy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import Metrics


def get_names(f_key):
    if f_key == Metrics.BVAE:
        names = ["BVAE_eval_accuracy"]  # , "BVAE_train_accuracy"]
    elif f_key == Metrics.RFVAE:
        names = ["RFVAE_eval_accuracy"]  # , "RFVAE_train_accuracy"]
    elif f_key == Metrics.FVAE:
        names = ["FVAE_eval_accuracy"]  # , "FVAE_train_accuracy"]
    elif f_key == Metrics.MODEX:
        names = ["MODEX_modularity_score", "MODEX_modularity_oldtest_score", "MODEX_modularity_oldtrain_score", "MODEX_explicitness_score_test"]  # "MODEX_explicitness_score_train",
    elif f_key == Metrics.DCIMIG:
        names = ["DCIMIG_normalized"]  # , "DCIMIG_unnormalized"]
    elif f_key == Metrics.DCI_RF_CLASS:
        names = ["DCI_RF_class_completeness", "DCI_RF_class_disentanglement", "DCI_RF_class_informativeness_test"]  # "DCI_RF_class_informativeness_train"]
    elif f_key == Metrics.DCI_RF_REG:
        names = ["DCI_RF_reg_completeness", "DCI_RF_reg_disentanglement", "DCI_RF_reg_informativeness_test"]  # , "DCI_RF_reg_informativeness_train"]
    elif f_key == Metrics.DCI_LOGREGL1:
        names = ["DCI_LogRegL1_completeness", "DCI_LogRegL1_disentanglement", "DCI_LogRegL1_informativeness_test"]  # , "DCI_LogRegL1_informativeness_train"]
    elif f_key == Metrics.DCI_LASSO:
        names = ["DCI_Lasso_completeness", "DCI_Lasso_disentanglement", "DCI_Lasso_informativeness_test"]  # , "DCI_Lasso_informativeness_train"]
    elif f_key == Metrics.SAP_DISCRETE:
        names = ["SAP_discrete"]  # , "SAP_discrete_train"]
    elif f_key == Metrics.JEMMIG:
        names = ["NORM_JEMMIG_score"]  # "JEMMIG_score",
    elif f_key == Metrics.MIG_SUP:
        names = ["MIG_sup_score"]
    elif f_key == Metrics.MIG:
        names = ["MIG_score"]
    elif f_key == Metrics.SAP_CONTINUOUS:
        names = ["SAP_continuous"]
    elif f_key == Metrics.IRS:
        names = ["IRS"]
    elif f_key == Metrics.WDG:
        names = ["WDG_score"]
    return names

# def make_table(results_dict, num_factors, val_per_factor):
#     #iterate through metrics
#     for f_key, K_dict in results_dict.items(): 
#          if f_key == Metrics.BVAE or f_key == Metrics.FVAE or f_key == Metrics.RFVAE:
             
#              alpha_dict = list(K_dict.values())[0]
#              batch_size_dict_sample = list(alpha_dict.values())[0]
#              num_eval_dict_sample = list(batch_size_dict_sample.values())[0]
             
#              for num_eval in num_evals:
#                     for batch_size in batch_sizes:
#         pass
        

def get_table_meanstd_info(K_dict, 
                           metric_names,
                           num_factors,
                           val_per_factor,
                           alphas,
                           orig_indexes=[],
                           indexes_name=[]):
    pd_indexes = {}
    rows_data = {}
    all_means = {}
    all_stds = {}
    for K, alpha_dict in K_dict.items():
        
        pd_indexes[K] = []
        rows_data[K] = []
        all_means[K] = []
        all_stds[K] = []
        
        for m_name in metric_names:
            
            indexes = orig_indexes.copy()
            indexes.append(m_name)
            
            all_scores = np.array([get_dict_element(d, indexes.copy()) for alpha_val, d in alpha_dict.items()])
            
            means = np.mean(all_scores, axis=1)
            all_means[K].append(means)
            
            stds = np.std(all_scores, axis=1)  
            all_stds[K].append(stds)
            
            mean_std_strs = ["{:.2f} ± {:.2f}".format(mean, std) for mean, std in zip(means, stds)]

            index_str = ""
            for index, index_name in zip(indexes[:-1], indexes_name):
                index_str = index_str + "{} = {} ".format(index_name, index)
            
            pd_indexes[K].append("{} {}".format(m_name,  index_str))
            rows_data[K].append(mean_std_strs)
           
    return rows_data, pd_indexes, all_means, all_stds     
    pass


def make_violon_plot(K_dict, metric_names, num_factors, val_per_factor, alphas, noise_mode, orig_indexes=[], indexes_name=[]):
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    for m_name in metric_names:
        indexes = orig_indexes.copy()
        indexes.append(m_name)
        
        labels = []  
        for K, alpha_dict in K_dict.items():
            all_scores = [get_dict_element(d, indexes.copy()) for alpha_val, d in alpha_dict.items()]
            add_label(plt.violinplot(all_scores, showmeans = True), "K = {}".format(K))      
         
            title_str = ""
            for index, index_name in zip(orig_indexes, indexes_name):
                title_str = title_str + "{}_{} ".format(index_name, index)

            plt.title("{}: Noise".format(m_name) + title_str +  "K = {}, {} Factors / {} values each".format(K, num_factors, val_per_factor))
            plt.xticks(range(1,len(alphas)+1), alphas)
            plt.xlabel("Noise-signal ratio")
            plt.ylabel(m_name)
            plt.legend(*zip(*labels), loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig('figs/{}/{}_{}'.format(str(noise_mode), m_name, title_str), bbox_inches='tight')
            plt.close()


def plot_mean_lines(means, mnames, alphas, K, noise_mode):
    for mean, mname in zip(means, mnames):
        plt.plot(alphas, mean, label=mname)
    plt.title("Mean values of metrics according to alpha, K={}".format(K))
    plt.xlabel("alpha")
    plt.ylabel("Metric Results") 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figs/{}/metric_means_K_{}.png'.format(str(noise_mode), K), bbox_inches='tight')
    plt.close()
    pass


def make_graphs(results_dict, num_factors, val_per_factor, noise_mode):

    if not os.path.exists('./figs/{}/'.format(str(noise_mode))):
        os.mkdir('./figs/{}/'.format(str(noise_mode)))

    # Define some necessary values found in dict.
    K_dict_dict_sample = list(results_dict.values())[0]
    alpha_dict_sample = list(K_dict_dict_sample.values())[0] 
    
    alphas = [alpha_val for alpha_val, __ in alpha_dict_sample.items()]
    Ks = [K_val for K_val, __ in K_dict_dict_sample.items()]
    
    all_means = {}
    all_names = {}
    tableDFs = {}
    for K in Ks:
        tableDFs[K] = pd.DataFrame(data=None, columns=alphas)
        all_means[K] = [] 
        all_names[K] = []

    # iterate through metrics
    for f_key, K_dict in results_dict.items(): 
        
        metric_names = get_names(f_key) 
        
        # These metrics have extra common parameters (batch-size, num_eval_train)
        if f_key == Metrics.BVAE or f_key == Metrics.FVAE or f_key == Metrics.RFVAE:       
            alpha_dict_sample = list(K_dict.values())[0] 
            batch_size_dict_sample = list(alpha_dict_sample.values())[0]
            num_eval_dict_sample = list(batch_size_dict_sample.values())[0]
            num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
            batch_sizes = [batch_size for batch_size, __ in batch_size_dict_sample.items()]

            # for batch_size in batch_sizes:
            #     for num_eval in num_evals:
            batch_size = batch_sizes[-1]  
            num_eval = num_evals[-1]  
            
            indexes = [batch_size, num_eval]
            index_names = ["Batch Size", "Num Eval"]
            make_violon_plot(K_dict, metric_names, num_factors, val_per_factor, alphas, noise_mode, indexes, index_names)
            rows_data, pd_indexes, means, stds = get_table_meanstd_info(K_dict, metric_names, num_factors, val_per_factor, alphas, indexes, index_names)
            
            # All functions append the mean-std table.
            for K in Ks:
                df = pd.DataFrame(data=rows_data[K], index=pd_indexes[K], columns=alphas)
                tableDFs[K] = tableDFs[K].append(df)
            
                for mean, name in zip(means[K], pd_indexes[K]):
                    all_means[K].append(mean)
                    all_names[K].append(name)
                    pass
            
        else:
            make_violon_plot(K_dict, metric_names, num_factors, val_per_factor, alphas, noise_mode)
            rows_data, pd_indexes, means, all_stds = get_table_meanstd_info(K_dict, metric_names, num_factors, val_per_factor, alphas)

            # All functions append the mean-std table.
            for K in Ks:
                df = pd.DataFrame(data=rows_data[K], index=pd_indexes[K], columns=alphas)
                tableDFs[K] = tableDFs[K].append(df)
            
                for mean, name in zip(means[K], pd_indexes[K]):
                    all_means[K].append(mean)
                    all_names[K].append(name)
                
    # Save table and fig for each K
    for K in Ks:    
        tableDFs[K].to_csv("figs/{}/big_table_K_{}.csv".format(str(noise_mode), K))
        all_means[K] = np.vstack(all_means[K])
        all_names[K] = np.vstack(all_names[K])
        plot_mean_lines(all_means[K], all_names[K], alphas, K, noise_mode)

    pass


def init_dict(d, all_params, depth):
    if len(all_params) > 0:
        params = all_params[0]
        all_params.remove(params)
        
        for p in params:
            d[p] = {}
            d[p] = init_dict(d[p], copy.copy(all_params), depth+1)
    return d


def add_to_dict(d, indexes_to_add, val, depth):
    if len(indexes_to_add) > 0:
        ind = indexes_to_add[0]
        indexes_to_add.remove(ind)
        d[ind] = add_to_dict(d[ind], indexes_to_add, val, depth+1)
    else:
        return val
    return d


def get_dict_element(d, indexes):
    if len(indexes) == 0:
        return d
    else:
        index = indexes[0]
        indexes.remove(index)
        return get_dict_element(d[index], indexes)


def manage_processes(processes, queue, max_process=10):
    """ @author: jboilard 
    from a list of already set processes, manage processes
    starts processes up to a certain maximum number of processes
    terminates the processes once they are over
    
    processes : already set processes from the function get_processes(...)
    results_dir : folder in which the process returns are saved
    queue : element in which the process outputs are saved
    """
    active_processes = []
    return_dicts = []
    while len(processes) > 0 or len(active_processes) > 0:
        # fill active processes list
        for process in processes:
            if process.is_alive() == False and process.exitcode == None and len(active_processes) < max_process: #process not started yet
                active_processes.append(process)
                active_processes[-1].start()
                          
        # check if any active_processes has ended
        ended_processes_idx = []
        for i, process in enumerate(active_processes):
            if process.is_alive() == False and process.exitcode == 0:  # process has ended
                print("No Error! {}".format(str(process.name)))
                process.terminate()
                processes.remove(process)
                ended_processes_idx.append(i)
                return_dicts.append(queue.get())

            elif process.is_alive() == False and process.exitcode == 1:  # process has ended
                print(str(process.name) + " ended with an error code : " + str(process.exitcode))
                process.terminate()
                processes.remove(process)
                ended_processes_idx.append(i)
         
        new_active_processes = []
        for i in range(len(active_processes)):
            if not i in ended_processes_idx:
                new_active_processes.append(active_processes[i])
        active_processes = new_active_processes

    return return_dicts
