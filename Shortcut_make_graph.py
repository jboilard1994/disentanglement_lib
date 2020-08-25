from disentanglement_lib.evaluation.benchmark.benchmark_utils import make_graphs
from disentanglement_lib.evaluation.benchmark.scenarios.scenario_noise import NoiseMode
import pickle

if __name__ == "__main__":

    # open a file, where you stored the pickled data
    file = open('./pickled_results/NoiseMode.FAV_CONTINUOUS_EXTRA_Z.p', 'rb')
    
    # dump information to that file
    [noise_mode, all_results] = pickle.load(file)
    
    # close the file
    file.close() 
    make_graphs(all_results, 3, 10, noise_mode)
