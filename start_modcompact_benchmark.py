import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigDCIRFClass
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigDCIRFReg
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigDCILogRegL1
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigDCILasso
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigSAPDiscrete
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigSAPContinuous

from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigBVAE
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigFVAE
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigRFVAE
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigIRS

from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigMIG
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigDCIMIG
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigMIGSUP
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigJEMMIG
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigModex
from disentanglement_lib.config.benchmark.scenarios.mod_compact_bindings import ConfigWDG

from disentanglement_lib.evaluation.benchmark.mod_compact_benchmark import mod_compact_scenario_main
from disentanglement_lib.evaluation.benchmark.scenarios.modcompact_dataholder import ModCompactMode
from disentanglement_lib.evaluation.benchmark.graphing.modcompact_graphing import make_graphs

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
TEST_BLUR = 1
TEST_CODE_FACTOR_DECAY = 2
TEST_MOD_MISSING_CHECK = 3
TEST_COMPACT_MISSING_CHECK = 4
TEST_TARGET_COMPACT_REDUCE = 5
TEST_MOD_REDUCE = 6
TEST_COMPACT_REDUCE = 7


mod_compact_modes = [ModCompactMode.TEST_BLUR,
                     ModCompactMode.TEST_CODE_FACTOR_DECAY,
                     ModCompactMode.TEST_MOD_MISSING_CHECK,
                     ModCompactMode.TEST_COMPACT_MISSING_CHECK,
                     ModCompactMode.TEST_MOD_REDUCE,
                     ModCompactMode.TEST_COMPACT_REDUCE]

if __name__ == "__main__": 
    process_mode = "mp"  # debug or mp
    num_factors = 3
    val_per_factor = 10
    n_seeds = 15

    for mod_compact_mode in mod_compact_modes:
        all_results = {}

        for f in config_funcs:
            results_dict = mod_compact_scenario_main(f, num_factors=num_factors,
                                                     val_per_factor=val_per_factor,
                                                     mod_compact_mode=mod_compact_mode,
                                                     nseeds=n_seeds,
                                                     process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

            if process_mode == "mp":
                pickle.dump([mod_compact_mode, all_results], open("./pickled_results/{}.p".format(str(mod_compact_mode)), "wb"))
                pass

        make_graphs(all_results, num_factors, val_per_factor, mod_compact_mode=mod_compact_mode)




