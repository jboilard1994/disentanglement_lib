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
from enum import Enum
from disentanglement_lib.evaluation.benchmark.metrics import beta_vae 
from disentanglement_lib.evaluation.benchmark.metrics import dci
from disentanglement_lib.evaluation.benchmark.metrics import factor_vae
from disentanglement_lib.evaluation.benchmark.metrics import dcimig
from disentanglement_lib.evaluation.benchmark.metrics import mig
from disentanglement_lib.evaluation.benchmark.metrics import wdg
from disentanglement_lib.evaluation.benchmark.metrics import jemmig
from disentanglement_lib.evaluation.benchmark.metrics import mig_sup
from disentanglement_lib.evaluation.benchmark.metrics import modularity_explicitness
from disentanglement_lib.evaluation.benchmark.metrics import sap_score
from disentanglement_lib.evaluation.benchmark.metrics import irs
from disentanglement_lib.evaluation.benchmark.metrics import rf_vae

class Metrics(Enum):
    DCI_RF_CLASS = 1
    DCI_RF_REG = 2
    DCI_LOGREGL1 = 3
    DCI_LASSO = 4
    BVAE = 5
    FVAE = 6
    RFVAE = 7
    MIG = 8
    WDG = 9
    DCIMIG = 10
    MIG_SUP = 11
    JEMMIG = 12
    MODEX = 13
    SAP_DISCRETE = 14
    SAP_CONTINUOUS = 15
    IRS = 16


class GenericConfigIRS:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get IRS configs, See Generic function description on top of file for more details"""
        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            discretizer_str = params_id[0]
            gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/irs.gin"]
            gin_bindings = ["irs.num_train = {}".format(n_samples),
                            "irs.batch_size = 16",
                            "discretizer.discretizer_fn = {}".format(discretizer_str),
                            "discretizer.num_bins = {}".format(val_per_factor)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer"]]
        param_ids = [["@histogram_discretizer"]]
        param_names = ["Discretizer Function"]
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id(): 
        return irs.compute_irs, Metrics.IRS


class GenericConfigSAPDiscrete:
    def get_gin_configs(self, n_samples, val_per_factor):
        return get_sap_configs(n_samples, val_per_factor, False)

    @staticmethod
    def get_extra_params():
        return [[]], [], []

    @staticmethod
    def get_metric_fn_id(): 
        return sap_score.compute_sap, Metrics.SAP_DISCRETE
    
    
class GenericConfigSAPContinuous:
    def get_gin_configs(self, n_samples, val_per_factor):
        return get_sap_configs(n_samples, val_per_factor, True)

    def get_extra_params(self):
        return [[]], [], []

    @staticmethod
    def get_metric_fn_id(): 
        return sap_score.compute_sap, Metrics.SAP_CONTINUOUS
    

def get_sap_configs(n_samples, val_per_factor, continuous): 
    """ Get SAP configs, See Generic function description on top of file for more details
    Extra args : continuous (bool) : defines the mode of the evaluated metric""" 
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/sap_score.gin"]
    gin_bindings = ["sap_score.num_train = {}".format(int(n_samples*0.8)),
                    "sap_score.num_test = {}".format(int(n_samples*0.2)),
                    "sap_score.continuous_factors = {}".format(continuous)]
    return [[gin_config_files, gin_bindings]]


class GenericConfigModex:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get MODEX configs, See Generic function description on top of file for more details"""
        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            discretizer_str = params_id[0]
            gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/modularity_explicitness.gin"]
            gin_bindings = ["modularity_explicitness.num_train = {}".format(int(n_samples*0.8)),
                            "modularity_explicitness.num_test = {}".format(int(n_samples*0.2)),
                            "discretizer.discretizer_fn = {}".format(discretizer_str),
                            "discretizer.num_bins = {}".format(val_per_factor)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer"]]
        param_ids = [["@histogram_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id(): 
        return modularity_explicitness.compute_modularity_explicitness, Metrics.MODEX
    

class GenericConfigRFVAE:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get FVAE configs, See Generic function description on top of file for more details
        Extra output params are 1) batch_size 2) num_train_evals """
        gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/rf_vae_metric.gin"]

        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            batch_size, num_train_eval = params_id
            gin_bindings = ["rf_vae_score.batch_size = {}".format(batch_size),
                            "rf_vae_score.num_train = {}".format(num_train_eval),
                            "rf_vae_score.num_eval = {}".format(num_train_eval)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        batch_sizes = [8]
        num_train_evals = [500]
        param_names = ["Batch Size", "Num generated data points"]
        extra_params = [batch_sizes, num_train_evals]
        param_ids = []
        for batch_size in batch_sizes:
            for num_train_eval in num_train_evals:
                param_ids.append([batch_size, num_train_eval])
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id():
        return rf_vae.compute_rf_vae, Metrics.RFVAE


class GenericConfigFVAE:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get FVAE configs, See Generic function description on top of file for more details
        Extra output params are 1) batch_size 2) num_train_evals """
        gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/factor_vae_metric.gin"]

        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            batch_size, num_train_eval = params_id
            gin_bindings = ["factor_vae_score.batch_size = {}".format(batch_size),
                            "factor_vae_score.num_train = {}".format(num_train_eval),
                            "factor_vae_score.num_eval = {}".format(num_train_eval)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        batch_sizes = [16]
        num_train_evals = [500]
        param_names = ["Batch Size", "Num generated data points"]
        extra_params = [batch_sizes, num_train_evals]
        param_ids = []
        for batch_size in batch_sizes:
            for num_train_eval in num_train_evals:
                param_ids.append([batch_size, num_train_eval])
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id():
        return factor_vae.compute_factor_vae, Metrics.FVAE


class GenericConfigBVAE:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get VVAE configs, See Generic function description on top of file for more details
        Extra output params are 1) batch_size 2) num_train_evals """
        gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/beta_vae_sklearn.gin"]

        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            batch_size, num_train_eval = params_id
            gin_bindings = ["beta_vae_sklearn.batch_size = {}".format(batch_size),
                            "beta_vae_sklearn.num_train = {}".format(num_train_eval),
                            "beta_vae_sklearn.num_eval = {}".format(num_train_eval)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        batch_sizes = [16]
        num_train_evals = [500]
        param_names = ["Batch Size", "Num generated data points"]
        extra_params = [batch_sizes, num_train_evals]
        param_ids = []
        for batch_size in batch_sizes:
            for num_train_eval in num_train_evals:
                param_ids.append([batch_size, num_train_eval])
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id():
        return beta_vae.compute_beta_vae_sklearn, Metrics.BVAE


class GenericConfigDCIRFClass:
    def get_gin_configs(self, n_samples, val_per_factor):
        return get_dci_configs(n_samples, val_per_factor, "RF_class")

    def get_extra_params(self):
        return [[]], [], []

    @staticmethod
    def get_metric_fn_id():
        return dci.compute_dci, Metrics.DCI_RF_CLASS


class GenericConfigDCIRFReg:
    def get_gin_configs(self, n_samples, val_per_factor):
        return get_dci_configs(n_samples, val_per_factor, "RF_reg")

    def get_extra_params(self):
        return [[]], [], []

    @staticmethod
    def get_metric_fn_id():
        return dci.compute_dci, Metrics.DCI_RF_REG


class GenericConfigDCILogRegL1:
    def get_gin_configs(self, n_samples, val_per_factor):
        return get_dci_configs(n_samples, val_per_factor, "LogRegL1")

    def get_extra_params(self):
        return [[]], [], []

    @staticmethod
    def get_metric_fn_id():
        return dci.compute_dci, Metrics.DCI_LOGREGL1


class GenericConfigDCILasso:
    def get_gin_configs(self, n_samples, val_per_factor):
        return get_dci_configs(n_samples, val_per_factor, "Lasso")

    def get_extra_params(self):
        return [[]], [], []

    @staticmethod
    def get_metric_fn_id(): 
        return dci.compute_dci, Metrics.DCI_LASSO


def get_dci_configs(n_samples, val_per_factor, mode): 
    """ Get SAP configs, See Generic function description on top of file for more details
    Extra args : continuous (bool) : defines the mode of the evaluated metric""" 
    gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/dci.gin"]
    gin_bindings = ["dci.num_train = {}".format(int(0.8*n_samples)),
                    "dci.num_eval = {}".format(int(n_samples*0.1)),
                    "dci.num_test = {}".format(int(n_samples*0.1)),
                    "dci.mode = '{}'".format(mode)]
    return [[gin_config_files, gin_bindings]]


class GenericConfigDCIMIG:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get DCIMIG configs, See Generic function description on top of file for more details"""
        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            discretizer_str = params_id[0]
            gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/dcimig.gin"]
            gin_bindings = ["dcimig.num_train = {}".format(n_samples),
                            "discretizer.discretizer_fn = {}".format(discretizer_str),
                            "discretizer.num_bins = {}".format(val_per_factor)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer"]]
        param_ids = [["@histogram_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id(): 
        return dcimig.compute_dcimig, Metrics.DCIMIG
    

class GenericConfigMIG:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get MIG configs, See Generic function description on top of file for more details"""
        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            discretizer_str = params_id[0]
            gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/mig.gin"]
            gin_bindings = ["mig.num_train = {}".format(n_samples),
                            "discretizer.discretizer_fn = {}".format(discretizer_str),
                            "discretizer.num_bins = {}".format(val_per_factor)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer"]]
        param_ids = [["@histogram_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id(): 
        return mig.compute_mig, Metrics.MIG
    

class GenericConfigWDG:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get MODEX configs, See Generic function description on top of file for more details"""
        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            discretizer_str = params_id[0]
            gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/wdg.gin"]
            gin_bindings = ["wdg.num_train = {}".format(n_samples),
                            "discretizer.discretizer_fn = {}".format(discretizer_str),
                            "discretizer.num_bins = {}".format(val_per_factor)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer"]]
        param_ids = [["@histogram_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id(): 
        return wdg.compute_wdg, Metrics.WDG
    

class GenericConfigJEMMIG:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get JEMMIG configs, See Generic function description on top of file for more details"""
        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            discretizer_str = params_id[0]
            gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/jemmig.gin"]
            gin_bindings = ["jemmig.num_train = {}".format(n_samples),
                            "discretizer.discretizer_fn = {}".format(discretizer_str),
                            "discretizer.num_bins = {}".format(val_per_factor)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer"]]
        param_ids = [["@histogram_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id():
        return jemmig.compute_jemmig, Metrics.JEMMIG
    

class GenericConfigMIGSUP:
    def get_gin_configs(self, n_samples, val_per_factor):
        """ Get MIGSUP configs, See Generic function description on top of file for more details"""
        configs = []
        params_ids = self.get_extra_params()[0]
        for params_id in params_ids:
            discretizer_str = params_id[0]
            gin_config_files = ["./disentanglement_lib/config/benchmark/metric_configs/mig_sup.gin"]
            gin_bindings = ["mig_sup.num_train = {}".format(n_samples),
                            "discretizer.discretizer_fn = {}".format(discretizer_str),
                            "discretizer.num_bins = {}".format(val_per_factor)]
            configs.append([gin_config_files, gin_bindings])
        return configs

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer"]]
        param_ids = [["@histogram_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names

    @staticmethod
    def get_metric_fn_id(): 
        return mig_sup.compute_mig_sup, Metrics.MIG_SUP
