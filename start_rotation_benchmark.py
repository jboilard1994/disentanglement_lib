import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

from disentanglement_lib.evaluation.benchmark.rotation_benchmark import rotation_scenario_main
from disentanglement_lib.evaluation.benchmark.scenarios.rotation_dataholder import RotationMode
from disentanglement_lib.evaluation.benchmark.graphing.rotation_graphing import make_graphs

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
    num_factors = 2
    val_per_factor = 50
    n_seeds = 20

    for rotation_mode in rotation_modes:
        all_results = {}

        for f in config_funcs:
            results_dict = rotation_scenario_main(f, num_factors=num_factors,
                                                  val_per_factor=val_per_factor,
                                                  rotation_mode=rotation_mode,
                                                  nseeds=n_seeds,
                                                  process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

            if process_mode == "mp":
                pickle.dump([rotation_mode, all_results], open("./pickled_results/{}.p".format(str(rotation_mode)), "wb"))
                pass

        make_graphs(all_results, num_factors, val_per_factor, rotation_mode=rotation_mode)




