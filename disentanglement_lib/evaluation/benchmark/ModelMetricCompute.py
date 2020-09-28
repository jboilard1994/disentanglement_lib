import gin
import numpy as np
from disentanglement_lib.evaluation.benchmark.scenarios.tensorflow_dataholder import TensorflowDataholder


def test_metric(discrete_factors, representations, index_dict, config_class, queue):
    # Get run parameters
    seed = index_dict["seed"]
    f = index_dict["f"]
    n_bins = 300

    config_ = config_class()

    # get params
    metric_fn = config_.get_metric_fn_id()[0]

    configs = config_.get_gin_configs(len(discrete_factors), n_bins)
    param_ids, all_params, param_names = config_.get_extra_params()

    results = []
    for i, config in enumerate(configs):
        gin_config_files, gin_bindings = config
        extra_param_id = param_ids[i]
        # apply configs
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)

        # set random states & go!
        random_state = np.random.RandomState(seed)

        dataholder = TensorflowDataholder(discrete_factors, representations)

        # Get scores and save in matrix
        score = metric_fn(dataholder, random_state)
        result_dict = {"seed": seed, "f": f, "score": score, "extra_params": extra_param_id, "param_names": param_names}
        results.append(result_dict)
        gin.clear_config()

    if not queue == None:
        queue.put(results)  # Multiprocessing accessible list.
    return results