import argparse
import pickle
import numpy as np
import warnings
import os
import sys
import multiprocessing as mp


warnings.simplefilter(action='ignore', category=FutureWarning)
from disentanglement_lib.data.ground_truth import util


def get_arguments():
    parser = argparse.ArgumentParser(description='scenario')
    parser.add_argument('--cwd', type=str, default='./',
                        help='Specify from which folder run scenarios')
    return parser.parse_args()


# change working directory for batch files.
args = get_arguments()
os.chdir(args.cwd)
sys.path.insert(0, args.cwd)


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

from disentanglement_lib.evaluation.benchmark.scenarios.tensorflow_dataholder import ModelMode
from disentanglement_lib.evaluation.benchmark.ModelMetricCompute import test_metric
from disentanglement_lib.evaluation.benchmark.benchmark_utils import manage_processes, init_dict, add_to_dict, organize_results
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.benchmark.graphing import alpha_parameterized_graphing
import gin

# We do not do SAP continuous, DCILasso and DCIRFREG because datasets don't have continuous factors.
config_funcs = [ConfigIRS,
                ConfigSAPDiscrete,
                ConfigModex,
                ConfigRFVAE,
                ConfigFVAE,
                ConfigBVAE,
                ConfigDCIRFClass,
                ConfigDCILogRegL1,
                ConfigDCIMIG,
                ConfigMIG,
                ConfigWDG,
                ConfigJEMMIG,
                ConfigMIGSUP]

config_funcs = [ConfigMIG,
                ConfigMIGSUP]

if __name__ == "__main__":
    params = []
    experiments_path = "D:\\projects\\disentanglement_lib_clear_eval\\output\\test_benchmark-experiment-6.1"
    metric_seeds = 4

    for dataset_name in os.listdir(experiments_path):
        alpha_strs = []
        all_results = {}

        for config_class in config_funcs:
            id_ = config_class.get_metric_fn_id()[1]
            all_results[id_] = []

        for alpha_str in os.listdir(os.path.join(experiments_path, dataset_name)):
            alpha_strs.append(alpha_str)
            factor_rep_path = os.path.join(experiments_path, dataset_name, alpha_str, "postprocessed", "sampled")

            factors = pickle.load(open( os.path.join(factor_rep_path, "factors.p"), "rb"))
            reps = pickle.load(open(os.path.join(factor_rep_path, "reps.p"), "rb"))

            for config_class in config_funcs:
                processes = []

                queue = mp.Queue()
                for seed in range(metric_seeds):
                    index_dict = {'K': 1, 'alpha': alpha_str, 'seed': seed, 'f': str(config_class.get_metric_fn_id()[0])}

                    process = mp.Process(target=test_metric,
                                         args=(factors,
                                               reps,
                                               index_dict,
                                               config_class,
                                               queue),
                                         name="seed = {}, fn={}".format(seed, str(config_class)))
                    processes.append(process)

                results_dict = manage_processes(processes, queue)
                organized_results = organize_results(results_dict, config_class)

                id_ = config_class.get_metric_fn_id()[1]
                all_results[id_].extend(organized_results)

        pickle.dump([ModelMode.BVAE, all_results], open("./pickled_results/{}_{}.p".format(str(ModelMode.BVAE), dataset_name), "wb"))
        #alpha_parameterized_graphing.make_graphs(all_results, factors.shape[1], -1, scenario_mode=ModelMode.BVAE)











