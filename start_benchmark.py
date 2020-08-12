
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_dci_L1
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_dci_gbt
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_bvae
from disentanglement_lib.evaluation.benchmark.benchmark import test_metric_fvae
from disentanglement_lib.evaluation.benchmark.benchmark import test_mig
from disentanglement_lib.evaluation.benchmark.benchmark import test_modex
from disentanglement_lib.evaluation.benchmark.benchmark import test_sap
from disentanglement_lib.evaluation.benchmark.benchmark import test_irs

#DOES NOT WORK YET
from disentanglement_lib.evaluation.benchmark.benchmark import test_fairness

from disentanglement_lib.evaluation.benchmark.benchmark import noise_experiment
from disentanglement_lib.evaluation.benchmark.benchmark import make_graphs

import pickle

#test_funcs =   [test_metric_dci_L1, 
#               test_metric_dci_gbt, 
#               test_metric_bvae, 
#               test_metric_fvae, 
#               test_mig, 
#               test_modex, 
#               test_sap, 
#               test_irs] 

test_funcs = [test_metric_dci_L1]  

if __name__ == "__main__":
    num_factors=3
    val_per_factor=10
    n_seeds = 50
    
    all_results = {}
    for f in test_funcs:
        
        results_dict = noise_experiment(f, num_factors=num_factors, val_per_factor=val_per_factor, nseeds = n_seeds)
        all_results[str(f)] = results_dict
        #pickle.dump(all_results, open( "results_dict.p", "wb" ))
          
    make_graphs(all_results, num_factors, val_per_factor)
