import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigDCIRFClass
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigDCIRFReg
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigDCILogRegL1
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigDCILasso
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigSAPDiscrete
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigSAPContinuous

from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigBVAE
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigFVAE
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigRFVAE
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigIRS

from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigMIG
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigDCIMIG
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigMIGSUP
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigJEMMIG
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigModex
from disentanglement_lib.config.benchmark.scenarios.nonlinear_bindings import ConfigWDG

from disentanglement_lib.evaluation.benchmark.nonlinear_benchmark import non_linear_scenario_main
from disentanglement_lib.evaluation.benchmark.scenarios.nonlinear_dataholder import NonlinearMode
from disentanglement_lib.evaluation.benchmark.graphing.non_linear_graphing import make_graphs


config_funcs = [ConfigRFVAE,
                ConfigFVAE,
                ConfigBVAE,
                ConfigIRS,
                ConfigSAPDiscrete,
                ConfigSAPContinuous,
                ConfigModex,
                ConfigDCIRFClass,
                ConfigDCIRFReg,
                ConfigDCILogRegL1,
                ConfigDCILasso,
                ConfigDCIMIG,
                ConfigMIG,
                ConfigWDG,
                ConfigJEMMIG,
                ConfigMIGSUP]

nonlinear_modes = [NonlinearMode.SIGMOID_FAV_CONTINUOUS,
                   NonlinearMode.SIGMOID_FAV_DISCRETE,
                   NonlinearMode.QUADRATIC_FAV_CONTINUOUS,
                   NonlinearMode.QUADRATIC_FAV_DISCRETE]

if __name__ == "__main__":
    num_factors = 3
    val_per_factor = 10 
    n_seeds = 10
    process_mode = "mp"  # "debug" or "mp" (multi-process)

    for nonlinear_mode in nonlinear_modes:
        all_results = {}

        for f in config_funcs:
            results_dict = non_linear_scenario_main(f, num_factors=num_factors,
                                                    val_per_factor=val_per_factor,
                                                    nonlinear_mode=nonlinear_mode,
                                                    nseeds=n_seeds,
                                                    process_mode=process_mode)

            id_ = f.get_metric_fn_id()[1]
            all_results[id_] = results_dict

        pickle.dump([nonlinear_mode, all_results], open("./pickled_results/{}.p".format(str(nonlinear_mode)), "wb"))
        make_graphs(all_results, num_factors, val_per_factor, nonlinear_mode=nonlinear_mode)
