import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from disentanglement_lib.evaluation.benchmark.benchmark_utils import init_dict, add_to_dict, get_dict_element, get_names
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics


def get_table_meanstd_info(theta_dict,
                           metric_names,
                           num_factors,
                           val_per_factor,
                           alphas,
                           orig_indexes=[],
                           indexes_name=[]):

    pd_indexes = []
    rows_data = []
    all_means = []
    all_stds = []

    for m_name in metric_names:

        indexes = orig_indexes.copy()
        indexes.append(m_name)

        all_scores = np.array([get_dict_element(d, indexes.copy()) for theta_val, d in theta_dict.items()])

        means = np.mean(all_scores, axis=1)
        all_means.append(means)

        stds = np.std(all_scores, axis=1)
        all_stds.append(stds)

        mean_std_strs = ["{:.2f} ± {:.2f}".format(mean, std) for mean, std in zip(means, stds)]

        index_str = ""
        for index, index_name in zip(indexes[:-1], indexes_name):
            index_str = index_str + "{} = {} ".format(index_name, index)

        pd_indexes.append("{} {}".format(m_name, index_str))
        rows_data.append(mean_std_strs)

    return rows_data, pd_indexes, all_means, all_stds
    pass


def make_violin_plot(theta_dict, metric_names, num_factors, val_per_factor, alphas, rotation_mode, orig_indexes=[],
                     indexes_name=[]):
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    for m_name in metric_names:
        indexes = orig_indexes.copy()
        indexes.append(m_name)

        labels = []
        all_scores = [get_dict_element(d, indexes.copy()) for __, d in theta_dict.items()]
        plt.violinplot(all_scores, showmeans=True)

        title_str = ""
        for index, index_name in zip(orig_indexes, indexes_name):
            title_str = title_str + "{}_{} ".format(index_name, index)

        plt.title(
            "{}: {}".format(m_name, str(rotation_mode)) + title_str + "{} Factors / {} values each".format(num_factors,
                                                                                                            val_per_factor))
        plt.xticks(range(1, len(alphas) + 1), alphas)
        plt.xlabel("Noise-signal ratio")
        plt.ylabel(m_name)
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 1))
        plt.savefig('figs/{}/{}_{}'.format(str(rotation_mode), m_name, title_str), bbox_inches='tight')
        plt.close()


def plot_mean_lines(means, mnames, thetas, rotation_mode):
    for mean, mname in zip(means, mnames):
        plt.plot(thetas, mean, label=mname)
    plt.title("Mean values of metrics according to theta")
    plt.xlabel("alpha")
    plt.ylabel("Metric Results")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figs/{}/metric_means.png'.format(str(rotation_mode)), bbox_inches='tight')
    plt.close()
    pass


def make_graphs(results_dict, num_factors, val_per_factor, rotation_mode):
    if not os.path.exists('./figs/{}/'.format(str(rotation_mode))):
        os.mkdir('./figs/{}/'.format(str(rotation_mode)))

    # Define some necessary values found in dict.
    theta_dict_sample = list(results_dict.values())[0]

    thetas = [theta_val for theta_val, __ in theta_dict_sample.items()]

    all_means = []
    all_names = []
    tableDF = pd.DataFrame(data=None, columns=thetas)

    # iterate through metrics
    for f_key, theta_dict in results_dict.items():

        metric_names = get_names(f_key)

        # These metrics have extra common parameters (batch-size, num_eval_train)
        if f_key == Metrics.BVAE or f_key == Metrics.FVAE or f_key == Metrics.RFVAE:
            batch_size_dict_sample = list(theta_dict.values())[0]
            num_eval_dict_sample = list(batch_size_dict_sample.values())[0]
            num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
            batch_sizes = [batch_size for batch_size, __ in batch_size_dict_sample.items()]

            # for batch_size in batch_sizes:
            #     for num_eval in num_evals:
            # Graph only biggest.
            batch_size = batch_sizes[-1]
            num_eval = num_evals[-1]

            indexes = [batch_size, num_eval]
            index_names = ["Batch Size", "Num Eval"]
            make_violin_plot(theta_dict, metric_names, num_factors, val_per_factor, thetas, rotation_mode, indexes,
                             index_names)
            rows_data, pd_indexes, means, stds = get_table_meanstd_info(theta_dict, metric_names, num_factors,
                                                                        val_per_factor, thetas, indexes, index_names)

            # All functions append the mean-std table.
            df = pd.DataFrame(data=rows_data, index=pd_indexes, columns=thetas)
            tableDF = tableDF.append(df)

            for mean, name in zip(means, pd_indexes):
                all_means.append(mean)
                all_names.append(name)
                pass

        else:
            make_violin_plot(theta_dict, metric_names, num_factors, val_per_factor, thetas, rotation_mode)
            rows_data, pd_indexes, means, all_stds = get_table_meanstd_info(theta_dict, metric_names, num_factors,
                                                                            val_per_factor, thetas)

            # All functions append the mean-std table.
            df = pd.DataFrame(data=rows_data, index=pd_indexes, columns=thetas)
            tableDF = tableDF.append(df)

            for mean, name in zip(means, pd_indexes):
                all_means.append(mean)
                all_names.append(name)
                pass

    tableDF.to_csv("figs/{}/big_table.csv".format(str(rotation_mode)))
    all_means = np.vstack(all_means)
    all_names = np.vstack(all_names)
    plot_mean_lines(all_means, all_names, thetas, rotation_mode)
    pass
