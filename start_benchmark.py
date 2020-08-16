
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_dci_RF_class
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_dci_RF_reg
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_dci_LogregL1
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_dci_Lasso

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_bvae
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_fvae

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_mig
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_mig_sup

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_modex

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_sap_discrete
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_sap_continuous

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import gin_irs

#from disentanglement_lib.evaluation.benchmark.benchmark import test_fairness

from disentanglement_lib.evaluation.benchmark.benchmark import noise_scenario_main
from disentanglement_lib.evaluation.benchmark.benchmark import make_graphs

import pickle

test_funcs = [gin_dci_RF_class, 
              gin_dci_RF_reg, 
              gin_dci_LogregL1,
              gin_dci_Lasso,
              gin_bvae,  
              gin_fvae, 
              gin_mig,
              gin_mig_sup,
              gin_modex,
              gin_sap_discrete, 
              gin_sap_continuous, 
              gin_irs] 


if __name__ == "__main__": 
    mode = "mp"
    num_factors = 3
    val_per_factor = 10
    n_seeds = 5
    
    all_results = {}
    for f in test_funcs:    
        results_dict = noise_scenario_main(f, num_factors=num_factors, val_per_factor=val_per_factor, nseeds = n_seeds, mode = mode)
        all_results[str(f)] = results_dict
        
        if mode == "mp":
            pickle.dump(all_results, open( "results_dict.p", "wb" ))
          
    make_graphs(all_results, num_factors, val_per_factor)
