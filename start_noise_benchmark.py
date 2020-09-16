import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigDCIRFClass
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigDCIRFReg
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigDCILogRegL1
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigDCILasso
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigSAPDiscrete
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigSAPContinuous

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigBVAE
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigFVAE
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigRFVAE
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigIRS

from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigMIG
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigDCIMIG
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigMIGSUP
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigJEMMIG
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigModex
from disentanglement_lib.config.benchmark.scenarios.noise_bindings import ConfigWDG

from disentanglement_lib.evaluation.benchmark.noise_benchmark import benchmark_main
from disentanglement_lib.evaluation.benchmark.scenarios.noise_dataholder import NoiseMode, NoiseDataHolder

from disentanglement_lib.evaluation.benchmark.graphing import alpha_parameterized_graphing

import pickle

config_funcs = [ConfigIRS,
                ConfigSAPDiscrete,
                ConfigSAPContinuous,
                ConfigModex,
                ConfigRFVAE,
                ConfigFVAE,
                ConfigBVAE,
                ConfigDCIRFClass,
                ConfigDCIRFReg,
                ConfigDCILogRegL1,
                ConfigDCILasso,
                ConfigDCIMIG,
                ConfigMIG,
                ConfigWDG,
                ConfigJEMMIG,
                ConfigMIGSUP]

noise_modes = [NoiseMode.NOISE_DECAY,
               NoiseMode.NOISE_DECAY_EXTRA_Z,
               NoiseMode.EXTRA_Z_COLLAPSED_TO_UNCOLLAPSED]

if __name__ == "__main__": 
    process_mode = "mp"  # debug or mp
    num_factors = 3
    val_per_factor = 10 
    n_seeds = 1

    # params
    ks = [1, 8]
    alphas = np.arange(0, 1.01, 0.2)
    alphas = [float("{:.2f}".format(a)) for a in alphas]

    for noise_mode in noise_modes:
        all_results = {}

        for f in config_funcs:
            results_dict = benchmark_main(dataholder_class=NoiseDataHolder,
                                          config_fn=f,
                                          num_factors=num_factors,
                                          val_per_factor=val_per_factor,
                                          scenario_mode=noise_mode,
                                          ks=ks,
                                          alphas=alphas,
                                          nseeds=n_seeds,
                                          process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

            pickle.dump([noise_mode, all_results], open("./pickled_results/b{}.p".format(str(noise_mode)), "wb"))

        alpha_parameterized_graphing.make_graphs(all_results, num_factors, val_per_factor, scenario_mode=noise_mode)




