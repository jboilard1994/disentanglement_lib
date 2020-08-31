from disentanglement_lib.evaluation.benchmark.graphing.non_linear_graphing import make_graphs as nl_make_graph
from disentanglement_lib.evaluation.benchmark.graphing.rotation_graphing import make_graphs as rotate_graph
from disentanglement_lib.evaluation.benchmark.graphing.noise_graphing import make_graphs as noise_make_graph
from disentanglement_lib.evaluation.benchmark.scenarios.noise_dataholder import NoiseMode
from disentanglement_lib.evaluation.benchmark.scenarios.nonlinear_dataholder import NonlinearMode
from disentanglement_lib.evaluation.benchmark.scenarios.rotation_dataholder import RotationMode
import pickle

if __name__ == "__main__":

    # open a file, where you stored the pickled data
    file = open('./pickled_results/NoiseMode.FAV_CONTINUOUS.p', 'rb')
    
    # dump information to that file
    [_mode, all_results] = pickle.load(file)
    
    # close the file
    file.close() 
    noise_make_graph(all_results, 3, 10, _mode)
