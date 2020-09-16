import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigDCIRFClass
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigDCIRFReg
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigDCILogRegL1
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigDCILasso
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigSAPDiscrete
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigSAPContinuous

from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigBVAE
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigFVAE
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigRFVAE
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigIRS

from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigMIG
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigDCIMIG
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigMIGSUP
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigJEMMIG
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigModex
from disentanglement_lib.config.benchmark.scenarios.rotation_bindings import ConfigWDG

from disentanglement_lib.evaluation.benchmark.noise_benchmark import benchmark_main
from disentanglement_lib.evaluation.benchmark.scenarios.rotation_dataholder import RotationMode, RotationDataHolder
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

rotation_modes = [RotationMode.CONTINUOUS]

if __name__ == "__main__": 
    process_mode = "mp"  # debug or mp
    num_factors = 4
    val_per_factor = 10
    n_seeds = 20

    thetas = np.arange(0, np.pi / 4 + np.pi / 20, np.pi / 20)
    thetas = [float("{:.2f}".format(a)) for a in thetas]

    for rotation_mode in rotation_modes:
        all_results = {}

        for f in config_funcs:
            results_dict = benchmark_main(dataholder_class=RotationDataHolder,
                                          config_fn=f,
                                          num_factors=num_factors,
                                          val_per_factor=val_per_factor,
                                          scenario_mode=rotation_mode,
                                          alphas=thetas,
                                          nseeds=n_seeds,
                                          process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

            if process_mode == "mp":
                pickle.dump([rotation_mode, all_results], open("./pickled_results/{}.p".format(str(rotation_mode)), "wb"))
                pass

        alpha_parameterized_graphing.make_graphs(all_results, num_factors, val_per_factor, rotation_mode)




