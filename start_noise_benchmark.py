import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_dci_RF_class
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_dci_RF_reg
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_dci_LogRegL1
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_dci_Lasso
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_sap_discrete
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_sap_continuous

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_bvae
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_fvae
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_rfvae
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_irs

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_mig
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_dcimig
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_mig_sup
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_jemmig
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_modex
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import config_wdg

from disentanglement_lib.evaluation.benchmark.noise_benchmark import noise_scenario_main
from disentanglement_lib.evaluation.benchmark.scenarios.noise_dataholder import NoiseMode
from disentanglement_lib.evaluation.benchmark.benchmark_utils import make_graphs

import pickle

config_funcs = [config_dci_RF_class, 
                config_dci_RF_reg, 
                config_dci_LogRegL1,
                config_dci_Lasso,
                config_bvae, 
                config_fvae, 
                config_rfvae,
                config_irs,
                config_mig, 
                config_wdg,
                config_dcimig,
                config_mig_sup,
                config_jemmig,
                config_modex,
                config_sap_discrete,
                config_sap_continuous]

noise_modes = [NoiseMode.FAV_CONTINUOUS,
               NoiseMode.FAV_CONTINUOUS_EXTRA_Z,
               NoiseMode.FAV_CONTINUOUS_SEEDED_DATASET,
               NoiseMode.FAV_DISCRETE,
               NoiseMode.FAV_DISCRETE_EXTRA_Z,
               NoiseMode.FAV_DISCRETE_SEEDED_DATASET,
               NoiseMode.FAV_DISCRETE_ADD_NOISE,
               NoiseMode.FAV_DISCRETE_ADD_NOISE_EXTRA_Z]

noise_modes = [NoiseMode.FAV_CONTINUOUS_EXTRA_Z]
config_funcs = [config_mig_sup, config_sap_discrete]


if __name__ == "__main__": 
    process_mode = "mp"  # debug or mp
    num_factors = 3
    val_per_factor = 10 
    n_seeds = 10

    for noise_mode in noise_modes:
        all_results = {}

        for f in config_funcs:
            results_dict = noise_scenario_main(f, num_factors=num_factors,
                                               val_per_factor=val_per_factor,
                                               noise_mode=noise_mode,
                                               nseeds=n_seeds,
                                               process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

            if process_mode == "mp":
               # pickle.dump([noise_mode, all_results], open("./pickled_results/{}.p".format(str(noise_mode)), "wb"))
                pass

        #make_graphs(all_results, num_factors, val_per_factor, noise_mode=noise_mode)




