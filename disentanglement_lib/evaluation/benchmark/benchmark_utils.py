# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:28:11 2020

@author: Jonathan Boilard
"""

import copy
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics


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
