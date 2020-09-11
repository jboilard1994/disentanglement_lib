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

from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigDCIRFClass
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigDCIRFReg
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigDCILogRegL1
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigDCILasso
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigSAPDiscrete
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigSAPContinuous

from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigBVAE
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigFVAE
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigRFVAE
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigIRS

from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigMIG
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigDCIMIG
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigMIGSUP
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigJEMMIG
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigModex
from disentanglement_lib.config.benchmark.scenarios.bindings import GenericConfigWDG


class ConfigIRS(GenericConfigIRS):
    def __init__(self):
        pass

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer", "@percentile_discretizer"]]
        param_ids = [["@histogram_discretizer"], ["@percentile_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names


class ConfigSAPDiscrete(GenericConfigSAPDiscrete):
    def __init__(self):
        pass


class ConfigSAPContinuous(GenericConfigSAPContinuous):
    def __init__(self):
        pass


class ConfigModex(GenericConfigModex):
    def __init__(self):
        pass

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer", "@percentile_discretizer"]]
        param_ids = [["@histogram_discretizer"], ["@percentile_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names


class ConfigRFVAE(GenericConfigRFVAE):
    def __init__(self):
        pass


class ConfigFVAE(GenericConfigFVAE):
    def __init__(self):
        pass


class ConfigBVAE(GenericConfigBVAE):
    def __init__(self):
        pass


class ConfigDCIRFClass(GenericConfigDCIRFClass):
    def __init__(self):
        pass


class ConfigDCIRFReg(GenericConfigDCIRFReg):
    def __init__(self):
        pass


class ConfigDCILogRegL1(GenericConfigDCILogRegL1):
    def __init__(self):
        pass


class ConfigDCILasso(GenericConfigDCILasso):
    def __init__(self):
        pass


class ConfigDCIMIG(GenericConfigDCIMIG):
    def __init__(self):
        pass

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer", "@percentile_discretizer"]]
        param_ids = [["@histogram_discretizer"], ["@percentile_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names


class ConfigMIG(GenericConfigMIG):
    def __init__(self):
        pass

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer", "@percentile_discretizer"]]
        param_ids = [["@histogram_discretizer"], ["@percentile_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names


class ConfigWDG(GenericConfigWDG):
    def __init__(self):
        pass

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer", "@percentile_discretizer"]]
        param_ids = [["@histogram_discretizer"], ["@percentile_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names


class ConfigJEMMIG(GenericConfigJEMMIG):
    def __init__(self):
        pass

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer", "@percentile_discretizer"]]
        param_ids = [["@histogram_discretizer"], ["@percentile_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names


class ConfigMIGSUP(GenericConfigMIGSUP):
    def __init__(self):
        pass

    def get_extra_params(self):
        extra_params = [["@histogram_discretizer", "@percentile_discretizer"]]
        param_ids = [["@histogram_discretizer"], ["@percentile_discretizer"]]
        param_names = [["Discretizer Function"]]
        return param_ids, extra_params, param_names


