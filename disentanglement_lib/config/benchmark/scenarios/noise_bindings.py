""" Author Jonathan Boilard 2020
Configs of all metrics relatively to the noise scenario.
For all functions :
    
Args:
    dataholder : DataHolder class objects which contains all factors/representations and all needed properties
  Returns:
    -List "configs":  
        [gin_files, --> Path to default metric configs
         gin_bindings, --> Custom configs adapted to the run
         metric_fn,  --> Metric function to evaluate 
         extra_params_index] --> index keys indicating where to concatenate the scores in the disctionnary.
                                 (If metric has no extra parameters to evaluate, extra_params_index = [])
                                 
    -List "all_extra_params" : For all extra parameters, list all possible unique values, 
                                used to initialize result dictionnary, if no extra params, 
                                all_extra_params = []"""
  
from disentanglement_lib.evaluation.benchmark.metrics import beta_vae 
from disentanglement_lib.evaluation.benchmark.metrics import dci
from disentanglement_lib.evaluation.benchmark.metrics import factor_vae
from disentanglement_lib.evaluation.benchmark.metrics import mig
from disentanglement_lib.evaluation.benchmark.metrics import mig_sup
from disentanglement_lib.evaluation.benchmark.metrics import modularity_explicitness
from disentanglement_lib.evaluation.benchmark.metrics import sap_score
from disentanglement_lib.evaluation.benchmark.metrics import irs

def gin_irs(dataholder): 
    """ Get IRS configs, See Generic function description on top of file for more details"""
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/irs.gin"]
    gin_bindings = ["irs.num_train = {}".format(len(dataholder.embed_codes)),
                    "irs.batch_size = 16",
                    "discretizer.discretizer_fn = @histogram_discretizer", 
                    "discretizer.num_bins = {}".format(dataholder.val_per_factor)]
    return [[gin_config_files, gin_bindings, irs.compute_irs, []]], []

def gin_sap_discrete(dataholder):
    """ Get configs for SAP in the discrete mode.
    Returns : configs of SAP in the continuous mode."""
    return gin_sap(dataholder, continuous=False)
    
def gin_sap_continuous(dataholder):
    """ Get configs for SAP in the discrete mode.
    Returns : configs of SAP in the continuous mode."""
    return gin_sap(dataholder, continuous=True)

def gin_sap(dataholder, continuous):   
    """ Get SAP configs, See Generic function description on top of file for more details
    Extra args : continuous (bool) : defines the mode of the evaluated metric""" 
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/sap_score.gin"]
    gin_bindings = ["sap_score.num_train = {}".format(int(0.8*len(dataholder.embed_codes))),
                    "sap_score.num_test = {}".format(int(len(dataholder.embed_codes)*0.2)),
                    "sap_score.continuous_factors = {}".format(continuous)]
    return [[gin_config_files, gin_bindings, sap_score.compute_sap, []]], []

def gin_modex(dataholder):   
    """ Get MODEX configs, See Generic function description on top of file for more details"""
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/modularity_explicitness.gin"]
    gin_bindings = ["modularity_explicitness.num_train = {}".format(int(0.8*len(dataholder.embed_codes))),
                    "modularity_explicitness.num_test = {}".format(int(len(dataholder.embed_codes)*0.2)),
                    "discretizer.discretizer_fn = @histogram_discretizer", 
                    "discretizer.num_bins = {}".format(dataholder.val_per_factor)]
    return [[gin_config_files, gin_bindings, modularity_explicitness.compute_modularity_explicitness, []]], []


def gin_fvae(dataholder):   
    """ Get FVAE configs, See Generic function description on top of file for more details
    Extra output params are 1)batch_size 2)num_train_evals """
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/factor_vae_metric.gin"]
    configs = []
    batch_size = 16
    num_train_evals = [50, 300, 500]
    extra_params = [[batch_size], num_train_evals]
    for num_train_eval in num_train_evals:
        gin_bindings = ["factor_vae_score.batch_size = {}".format(16),
                        "factor_vae_score.num_train = {}".format(num_train_eval),
                        "factor_vae_score.num_eval = {}".format(num_train_eval)]
        add_ids = [batch_size, num_train_eval]
        configs.append([gin_config_files, gin_bindings, factor_vae.compute_factor_vae, add_ids])     
    return configs, extra_params

def gin_bvae(dataholder):   
    """ Get BVAE configs, See Generic function description on top of file for more details
    Extra output params are 1)batch_size 2)num_train_evals """
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/beta_vae_sklearn.gin"]
    configs = []
    batch_size = 16
    num_train_evals = [50, 300, 500]
    extra_params = [[batch_size], num_train_evals]
    for num_train_eval in num_train_evals:
        gin_bindings = ["beta_vae_sklearn.batch_size = {}".format(16),
                        "beta_vae_sklearn.num_train = {}".format(num_train_eval),
                        "beta_vae_sklearn.num_eval = {}".format(num_train_eval)]
        add_ids = [batch_size, num_train_eval]
        configs.append([gin_config_files, gin_bindings, beta_vae.compute_beta_vae_sklearn, add_ids])       
    return configs, extra_params
        
def gin_dci_RF_class(dataholder):
    """ Get configs for DCI using Random Forest classifier.
    Returns : configs of DCI in Random Forest classification mode."""
    return gin_dci(dataholder, "RF_class")
    
def gin_dci_RF_reg(dataholder):
    """ Get configs for DCI using Random Forest regressor.
    Returns : configs of DCI in Random Forest regression mode."""
    return gin_dci(dataholder, "RF_reg")

def gin_dci_LogregL1(dataholder):
    """ Get configs for DCI using L1 penalty Logistic regression classifier.
    Returns : configs of DCI in L1 penalty Logistic regression classification mode."""
    return gin_dci(dataholder, "LogregL1")
    
def gin_dci_Lasso(dataholder):
    """ Get configs for DCI using Lasso regressor.
    Returns : configs of DCI in Lasso regression mode."""
    return gin_dci(dataholder, "Lasso")

def gin_dci(dataholder, mode):
    """ Get DCI configs, See Generic function description on top of file for more details
    Extra args : mode (str) : defines the sklearn model to use in the evaluated metric"""
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/dci.gin"]
    gin_bindings = ["dci.num_train = {}".format(int(0.8*len(dataholder.embed_codes))),
                    "dci.num_eval = {}".format(int(len(dataholder.embed_codes)*0.1)),
                    "dci.num_test = {}".format(int(len(dataholder.embed_codes)*0.1)),
                    "dci.mode = '{}'".format(mode)]
    return [[gin_config_files, gin_bindings, dci.compute_dci, []]], []

def gin_mig(dataholder):   
    """ Get MIG configs, See Generic function description on top of file for more details"""
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/mig.gin"]
    gin_bindings = ["mig.num_train = {}".format(len(dataholder.embed_codes)),
                    "discretizer.discretizer_fn = @histogram_discretizer",
                    "discretizer.num_bins = {}".format(dataholder.val_per_factor)]
    return [[gin_config_files, gin_bindings, mig.compute_mig, []]], []

def gin_mig_sup(dataholder): 
    """ Get MIG-sup configs, See Generic function description on top of file for more details"""
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/mig_sup.gin"]
     
    gin_bindings = ["mig.num_train = {}".format(len(dataholder.embed_codes)),
                    "discretizer.discretizer_fn = @histogram_discretizer",
                    "discretizer.num_bins = {}".format(dataholder.val_per_factor)]
    return [[gin_config_files, gin_bindings, mig_sup.compute_mig_sup, []]], []