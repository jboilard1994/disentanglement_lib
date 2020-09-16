import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from disentanglement_lib.evaluation.benchmark.benchmark_utils import init_dict, add_to_dict, get_dict_element, get_names
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics


def get_table_meanstd_info(fn_results_list,
                           metric_names,
                           num_factors,
                           val_per_factor):
    pd_indexes = {}
    rows_data = {}
    all_means = {}
    all_stds = {}
    grouped_dict = group_list_in_dict(fn_results_list)

    for k in grouped_dict["unique_ks"]:
        pd_indexes[k] = []
        rows_data[k] = []
        all_means[k] = []
        all_stds[k] = []

    for m_name in metric_names:
        for extra_params in grouped_dict["extra_params_unique"]:
            param_ids = np.argwhere((grouped_dict["extra_params"] == extra_params).all(-1))
            param_list = np.take(fn_results_list, param_ids, axis=0).flatten()
            param_dict = group_list_in_dict(param_list)

            for k in grouped_dict["unique_ks"]:
                k_ids = np.argwhere(param_dict["ks"] == k)
                k_param_list = np.take(param_list, k_ids, axis=0).flatten()
                k_param_dict = group_list_in_dict(k_param_list)

                alphas_score_list = []
                for alpha in grouped_dict["unique_alphas"]:
                    alpha_ids = np.argwhere(k_param_dict["alphas"] == alpha)
                    alpha_param_k_list = np.take(k_param_list, alpha_ids, axis=0).flatten()
                    scores = [dict_["score"][m_name] for dict_ in alpha_param_k_list]
                    alphas_score_list.append(scores)

                means = np.mean(alphas_score_list, axis=1)
                all_means[k].append(means)

                stds = np.std(alphas_score_list, axis=1)
                all_stds[k].append(stds)

                mean_std_strs = ["{:.2f} ± {:.2f}".format(mean, std) for mean, std in zip(means, stds)]

                index_str = ""
                for index, index_name in zip(extra_params, grouped_dict["param_names"]):
                    index_str = index_str + "{} = {} ".format(index_name, index)

                pd_indexes[k].append("{} {}".format(m_name, index_str))
                rows_data[k].append(mean_std_strs)

    return rows_data, pd_indexes, all_means, all_stds


def plot_mean_lines(means, mnames, alphas, K, scenario_mode, mode="all"):
    linestyle_tuple = [
        (0, ()),       # 'solid',
        (0, (1, 5)),   # loosely dotted
        (0, (1, 1)),   # dotted

        (0, (3, 10)),  # loosely dashed
        (0, (6, 3)),   # dashed
        (0, (10, 1)),  # densely dashed

        (0, (5, 10, 1, 10)),   # loosely dashdotted
        (0, (3, 3, 1, 5)),     # dashdotted
        (0, (3, 1, 1, 1)),     # densely dashdotted

        (0, (3, 5, 1, 5, 1, 5)),     # dashdotdotted
        (0, (10, 15, 1, 15, 1, 1)),  # loosely dashdotdotted
        (0, (10, 1, 1, 1, 1, 1)),     # densely dashdotdotted

        (0, (15, 5)),   # broadly dashed
        (0, (30, 1)),   # Tight broadly dashed
        (0, (15, 15)),  # broadly loosely dashed

        (0, (10, 1, 1, 2)),   # Getting Desperate
        (0, (1, 2, 10, 5))]   # Getting Desperate 2
    linestyle_tuple = linestyle_tuple*3

    n_lines = len(means)

    for mean, mname, linestyle in zip(means, mnames, linestyle_tuple[:n_lines]):
        plt.plot(alphas, mean, label=mname, linestyle=linestyle)

    #plt.title("Mean values of metrics according to alpha, K={}".format(K))
    plt.xlabel("⍺")
    plt.xticks(alphas)
    plt.ylabel("Mean Score")
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1.05))
    plt.legend(handlelength=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figs/{}/metric_means_K_{}_{}.png'.format(str(scenario_mode), K, mode), bbox_inches='tight')
    plt.close()
    pass

def group_list_in_dict(results_dict_list):
    grouped_dict = {}
    grouped_dict["ks"] = [dict_["K"] for dict_ in results_dict_list]
    grouped_dict["alphas"] = [dict_["alpha"] for dict_ in results_dict_list]
    grouped_dict["extra_params"] = [dict_["extra_params"] for dict_ in results_dict_list]
    grouped_dict["scores"] = [dict_["score"] for dict_ in results_dict_list]

    grouped_dict["unique_ks"] = np.unique(grouped_dict["ks"])
    grouped_dict["param_names"] = results_dict_list[0]["param_names"]
    grouped_dict["unique_alphas"] = np.unique(grouped_dict["alphas"])
    grouped_dict["extra_params_unique"] = np.unique(grouped_dict["extra_params"], axis=0)

    grouped_dict["seeds"] = [dict_["seed"] for dict_ in results_dict_list]
    grouped_dict["n_seeds"] = np.max(grouped_dict["seeds"]) + 1
    return grouped_dict

def get_list_dict_from_ids(_list_to_sample, ids):
    _list = np.take(_list_to_sample, ids, axis=0).flatten()
    _dict = group_list_in_dict(_list)
    return _list, _dict


def make_violin_plot(fn_results_list, metric_names, num_factors, val_per_factor, scenario_mode):
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    grouped_dict = group_list_in_dict(fn_results_list)
    for m_name in metric_names:
        for extra_params in grouped_dict["extra_params_unique"]:
            labels = []
            param_list, param_dict = get_list_dict_from_ids(fn_results_list, np.argwhere((grouped_dict["extra_params"] == extra_params).all(-1)))

            for k in grouped_dict["unique_ks"]:
                k_param_list, k_param_dict = get_list_dict_from_ids(param_list, np.argwhere(param_dict["ks"] == k))

                alphas_score_list = []
                for alpha in grouped_dict["unique_alphas"]:
                    alpha_k_param_list, alpha_k_param_dict = get_list_dict_from_ids(k_param_list, np.argwhere(k_param_dict["alphas"] == alpha))
                    scores = [dict_["score"][m_name] for dict_ in alpha_k_param_list]
                    alphas_score_list.append(scores)

                if len(grouped_dict["unique_ks"]) == 1:
                    plt.violinplot(alphas_score_list, showmeans=True)
                else:
                    add_label(plt.violinplot(alphas_score_list, showmeans=True), "K = {}".format(k))

            title_str = ""
            for extra_param, param_name in zip(extra_params, grouped_dict["param_names"]):
                title_str = title_str + "{}_{} ".format(param_name, extra_param)

            plt.title("{}; Mode: {}".format(m_name, str(scenario_mode))) #+ title_str + "{} Factors / {} values each".format(num_factors, val_per_factor))
            plt.xticks(range(1, len(grouped_dict["unique_alphas"]) + 1), grouped_dict["unique_alphas"])
            plt.xlabel("⍺")
            plt.ylabel(m_name)

            if len(grouped_dict["unique_ks"]) > 1:
                plt.legend(*zip(*labels), loc='center left', bbox_to_anchor=(1, 0.5))

            x1, x2, y1, y2 = plt.axis()
            plt.axis((x1, x2, 0, 1.05))
            plt.savefig('figs/{}/{}_{}'.format(str(scenario_mode), m_name, title_str), bbox_inches='tight')
            plt.close()


def make_graphs(results_dict_list, num_factors, val_per_factor, scenario_mode):
    if not os.path.exists('./figs/{}/'.format(str(scenario_mode))):
        os.mkdir('./figs/{}/'.format(str(scenario_mode)))
    all_means = {}
    all_names = {}
    all_means_2 = {}
    all_names_2 = {}
    sample_dict = group_list_in_dict(list(results_dict_list.values())[0])

    tableDFs = {}
    for k in sample_dict["unique_ks"]:
        tableDFs[k] = pd.DataFrame(data=None, columns=sample_dict["unique_alphas"])
        all_means[k] = []
        all_names[k] = []
        all_means_2[k] = []
        all_names_2[k] = []

    for f_key, fn_results_list in results_dict_list.items():
        grouped_dict = group_list_in_dict(fn_results_list)
        metric_names = get_names(f_key, mode="")
        metric_parsed_names = get_names(f_key, mode="parsed")

        if len(metric_names) > 0:
            make_violin_plot(fn_results_list, metric_names, num_factors, val_per_factor, scenario_mode)

            rows_data, pd_indexes, means, stds = get_table_meanstd_info(fn_results_list, metric_names, num_factors,val_per_factor)
            rows_data_2, pd_indexes_2, means_2, stds_2 = get_table_meanstd_info(fn_results_list, metric_parsed_names, num_factors,
                                                                                val_per_factor)

            # All functions append the mean-std table.
            for K in grouped_dict["unique_ks"]:
                df = pd.DataFrame(data=rows_data[K], index=pd_indexes[K], columns=grouped_dict["unique_alphas"])
                tableDFs[K] = tableDFs[K].append(df)

                for mean, name in zip(means[K], pd_indexes[K]):
                    all_means[K].append(mean)
                    all_names[K].append(name)
                    pass

                #Show only one hyperparameter for clean version.
                if f_key == Metrics.FVAE or f_key == Metrics.RFVAE or f_key == Metrics.BVAE:
                    means_2[K] = [means_2[K][-1]]
                    pd_indexes_2[K] = [pd_indexes_2[K][-1]]

                for mean, name in zip(means_2[K], pd_indexes_2[K]):
                    all_means_2[K].append(mean)
                    all_names_2[K].append(name)
                    pass

    # Save table and fig for each K
    for K in grouped_dict["unique_ks"]:
        tableDFs[K].to_csv("figs/{}/big_table_K_{}.csv".format(str(scenario_mode), K))
        all_means[K] = np.vstack(all_means[K])
        all_names[K] = np.vstack(all_names[K])
        all_means_2[K] = np.vstack(all_means_2[K])
        all_names_2[K] = np.vstack(all_names_2[K])

        plot_mean_lines(all_means[K], all_names[K], grouped_dict["unique_alphas"], K, scenario_mode, mode="all")
        plot_mean_lines(all_means_2[K], all_names_2[K], grouped_dict["unique_alphas"], K, scenario_mode, mode="parsed")

