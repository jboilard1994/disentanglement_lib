
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_dci_RF_class
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_dci_RF_reg
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_dci_LogregL1
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_dci_Lasso

from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_bvae
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_fvae
from disentanglement_lib.evaluation.benchmark.benchmark import test_mig
from disentanglement_lib.evaluation.benchmark.benchmark import test_modex
from disentanglement_lib.evaluation.benchmark.benchmark import test_sap_discrete
from disentanglement_lib.evaluation.benchmark.benchmark import test_sap_continuous
from disentanglement_lib.evaluation.benchmark.benchmark import test_irs

#DOES NOT WORK YET
from disentanglement_lib.evaluation.benchmark.benchmark import test_fairness

from disentanglement_lib.evaluation.benchmark.benchmark import noise_experiment
from disentanglement_lib.evaluation.benchmark.benchmark import make_graphs

import pickle

# test_funcs =   [test_metric_dci_RF_class, 
#               test_metric_dci_RF_reg, 
#               test_metric_dci_LogregL1,
#               test_metric_dci_Lasso,
#               test_metric_bvae,  
#               test_metric_fvae, 
#               test_mig, 
#               test_modex, 
#               test_sap_discrete, 
#               test_sap_continuous, 
#               test_irs] 

test_funcs = [test_metric_bvae]


if __name__ == "__main__": 
    num_factors = 3
    val_per_factor = 10
    n_seeds = 2
    
    all_results = {}
    for f in test_funcs:
        
        results_dict = noise_experiment(f, num_factors=num_factors, val_per_factor=val_per_factor, nseeds = n_seeds)
        all_results[str(f)] = results_dict
        #pickle.dump(all_results, open( "results_dict.p", "wb" ))
          
    make_graphs(all_results, num_factors, val_per_factor)
