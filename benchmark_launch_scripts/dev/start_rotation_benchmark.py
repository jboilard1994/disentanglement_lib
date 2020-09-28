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
from disentanglement_lib.evaluation.benchmark.graphing import parameter_free_graphing

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


    alphas = [0]

    for rotation_mode in rotation_modes:
        all_results = {}

        for f in config_funcs:
            results_dict = benchmark_main(dataholder_class=RotationDataHolder,
                                          config_fn=f,
                                          num_factors=num_factors,
                                          val_per_factor=val_per_factor,
                                          scenario_mode=rotation_mode,
                                          alphas=alphas,
                                          nseeds=n_seeds,
                                          process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

            if process_mode == "mp":
                pickle.dump([rotation_mode, all_results], open("./pickled_results/{}.p".format(str(rotation_mode)), "wb"))
                pass

        parameter_free_graphing.make_graphs(all_results, num_factors, val_per_factor, rotation_mode)




