from disentanglement_lib.evaluation.benchmark.graphing.non_linear_graphing import make_graphs as nl_make_graph
from disentanglement_lib.evaluation.benchmark.graphing.rotation_graphing import make_graphs as rotate_graph
from disentanglement_lib.evaluation.benchmark.graphing.noise_graphing import make_graphs as noise_make_graph
from disentanglement_lib.evaluation.benchmark.graphing.modcompact_graphing import make_graphs as mc_make_graph

from disentanglement_lib.evaluation.benchmark.scenarios.noise_dataholder import NoiseMode
from disentanglement_lib.evaluation.benchmark.scenarios.nonlinear_dataholder import NonlinearMode
from disentanglement_lib.evaluation.benchmark.scenarios.rotation_dataholder import RotationMode
from disentanglement_lib.evaluation.benchmark.scenarios.modcompact_dataholder import ModCompactMode

import pickle
import os



#path = "./pickled_results/"
path = "./pickled_results"
file_list = os.listdir(path)

for file in file_list:
    filepath = os.path.join(path, file)
    if os.path.isfile(filepath):
        # open a file, where you stored the pickled data
        file = open(filepath, 'rb')

        # dump information to that file
        [_mode, all_results] = pickle.load(file)

        if _mode in NoiseMode:
            noise_make_graph(all_results, 3, 10, _mode)
        elif _mode in NonlinearMode:
            nl_make_graph(all_results, 3, 10, _mode)
        elif _mode in RotationMode:
            rotate_graph(all_results, 3, 10, _mode)
        elif _mode in ModCompactMode:
            mc_make_graph(all_results, 3, 10, _mode)

        # close the file
        file.close()

