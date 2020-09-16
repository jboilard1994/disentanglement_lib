import argparse
import pickle
import numpy as np
import warnings
import os
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_arguments():
    parser = argparse.ArgumentParser(description='scenario')
    parser.add_argument('--cwd', type=str, default='./',
                        help='Specify from which folder run scenarios')
    return parser.parse_args()


# change working directory for batch files.
args = get_arguments()
os.chdir(args.cwd)
sys.path.insert(0, args.cwd)


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

from disentanglement_lib.evaluation.benchmark.noise_benchmark import benchmark_main
from disentanglement_lib.evaluation.benchmark.scenarios.modcompact_dataholder import ModCompactMode, ModCompactDataHolder

from disentanglement_lib.evaluation.benchmark.graphing import alpha_parameterized_graphing
from disentanglement_lib.evaluation.benchmark.graphing import parameter_free_graphing





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

""" mod_compact_modes = [ModCompactMode.TEST_BLUR,
                     ModCompactMode.TEST_CODE_FACTOR_DECAY,
                     ModCompactMode.TEST_MOD_MISSING_CHECK,
                     ModCompactMode.TEST_COMPACT_MISSING_CHECK,
                     ModCompactMode.TEST_MOD_REDUCE,
                     ModCompactMode.TEST_COMPACT_REDUCE]"""

mod_compact_modes = [ModCompactMode.TEST_MOD_MISSING_CHECK,
                     ModCompactMode.TEST_COMPACT_MISSING_CHECK,
                     ModCompactMode.TEST_COMPACT_REDUCE]

if __name__ == "__main__": 
    process_mode = "mp"  # debug or mp
    num_factors = 3
    val_per_factor = 10
    n_seeds = 20


    for mod_compact_mode in mod_compact_modes:
        if not mod_compact_mode == ModCompactMode.TEST_COMPACT_MISSING_CHECK and not mod_compact_mode == ModCompactMode.TEST_MOD_MISSING_CHECK:
            alphas = np.arange(0, 1.01, 0.2)
            alphas = [float("{:.2f}".format(a)) for a in alphas]
        else:
            alphas = [0]

        all_results = {}

        for f in config_funcs:
            results_dict = benchmark_main(dataholder_class=ModCompactDataHolder,
                                          config_fn=f,
                                          num_factors=num_factors,
                                          val_per_factor=val_per_factor,
                                          scenario_mode=mod_compact_mode,
                                          alphas=alphas,
                                          nseeds=n_seeds,
                                          process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

            if process_mode == "mp":
                pickle.dump([mod_compact_mode, all_results], open("./pickled_results/{}.p".format(str(mod_compact_mode)), "wb"))
                pass

        if mod_compact_mode == ModCompactMode.TEST_MOD_MISSING_CHECK or mod_compact_mode == ModCompactMode.TEST_COMPACT_MISSING_CHECK:
            parameter_free_graphing.make_graphs(all_results, num_factors, val_per_factor, mod_compact_mode)
        else:
            if mod_compact_mode == ModCompactMode.TEST_MOD_REDUCE:
                num_factors = num_factors * 2
            alpha_parameterized_graphing.make_graphs(all_results, num_factors, val_per_factor, mod_compact_mode)




