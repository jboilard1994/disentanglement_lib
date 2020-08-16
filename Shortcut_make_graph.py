from disentanglement_lib.evaluation.benchmark.benchmark import make_graphs
import pickle



if __name__ == "__main__":

    #pickle.dump(all_results, open( "results_dict.p", "wb" ))
    # open a file, where you stored the pickled data
    file = open('results_dict.p', 'rb')
    
    # dump information to that file
    all_results = pickle.load(file)
    
    # close the file
    file.close() 
    make_graphs(all_results, 3, 10)
