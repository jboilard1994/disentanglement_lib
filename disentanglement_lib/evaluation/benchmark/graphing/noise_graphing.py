import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from disentanglement_lib.evaluation.benchmark.benchmark_utils import init_dict, add_to_dict, get_dict_element, get_names
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics


def get_table_meanstd_info(K_dict,
                           metric_names,
                           num_factors,
                           val_per_factor,
                           alphas,
                           orig_indexes=[],
                           indexes_name=[]):
    pd_indexes = {}
    rows_data = {}
    all_means = {}
    all_stds = {}
    for K, alpha_dict in K_dict.items():

        pd_indexes[K] = []
        rows_data[K] = []
        all_means[K] = []
        all_stds[K] = []

        for m_name in metric_names:

            indexes = orig_indexes.copy()
            indexes.append(m_name)

            all_scores = np.array([get_dict_element(d, indexes.copy()) for alpha_val, d in alpha_dict.items()])

            means = np.mean(all_scores, axis=1)
            all_means[K].append(means)

            stds = np.std(all_scores, axis=1)
            all_stds[K].append(stds)

            mean_std_strs = ["{:.2f} ± {:.2f}".format(mean, std) for mean, std in zip(means, stds)]

            index_str = ""
            for index, index_name in zip(indexes[:-1], indexes_name):
                index_str = index_str + "{} = {} ".format(index_name, index)

            pd_indexes[K].append("{} {}".format(m_name, index_str))
            rows_data[K].append(mean_std_strs)

    return rows_data, pd_indexes, all_means, all_stds
    pass


def make_violin_plot(K_dict, metric_names, num_factors, val_per_factor, alphas, noise_mode, orig_indexes=[],
                     indexes_name=[]):
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    for m_name in metric_names:
        indexes = orig_indexes.copy()
        indexes.append(m_name)

        labels = []
        for K, alpha_dict in K_dict.items():
            all_scores = [get_dict_element(d, indexes.copy()) for alpha_val, d in alpha_dict.items()]
            add_label(plt.violinplot(all_scores, showmeans=True), "K = {}".format(K))

            title_str = ""
            for index, index_name in zip(orig_indexes, indexes_name):
                title_str = title_str + "{}_{} ".format(index_name, index)

            plt.title(
                "{}: Noise".format(m_name) + title_str + "K = {}, {} Factors / {} values each".format(K, num_factors,
                                                                                                      val_per_factor))
            plt.xticks(range(1, len(alphas) + 1), alphas)
            plt.xlabel("Noise-signal ratio")
            plt.ylabel(m_name)
            plt.legend(*zip(*labels), loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig('figs/{}/{}_{}'.format(str(noise_mode), m_name, title_str), bbox_inches='tight')
            plt.close()


def plot_mean_lines(means, mnames, alphas, K, noise_mode):
    for mean, mname in zip(means, mnames):
        plt.plot(alphas, mean, label=mname)
    plt.title("Mean values of metrics according to alpha, K={}".format(K))
    plt.xlabel("alpha")
    plt.ylabel("Metric Results")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figs/{}/metric_means_K_{}.png'.format(str(noise_mode), K), bbox_inches='tight')
    plt.close()
    pass

def make_graphs(results_dict, num_factors, val_per_factor, noise_mode):
    if not os.path.exists('./figs/{}/'.format(str(noise_mode))):
        os.mkdir('./figs/{}/'.format(str(noise_mode)))

    # Define some necessary values found in dict.
    K_dict_dict_sample = list(results_dict.values())[0]
    alpha_dict_sample = list(K_dict_dict_sample.values())[0]

    alphas = [alpha_val for alpha_val, __ in alpha_dict_sample.items()]
    Ks = [K_val for K_val, __ in K_dict_dict_sample.items()]

    all_means = {}
    all_names = {}
    tableDFs = {}
    for K in Ks:
        tableDFs[K] = pd.DataFrame(data=None, columns=alphas)
        all_means[K] = []
        all_names[K] = []

    # iterate through metrics
    for f_key, K_dict in results_dict.items():

        metric_names = get_names(f_key)

        # These metrics have extra common parameters (batch-size, num_eval_train)
        if f_key == Metrics.BVAE or f_key == Metrics.FVAE or f_key == Metrics.RFVAE:
            alpha_dict_sample = list(K_dict.values())[0]
            batch_size_dict_sample = list(alpha_dict_sample.values())[0]
            num_eval_dict_sample = list(batch_size_dict_sample.values())[0]
            num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
            batch_sizes = [batch_size for batch_size, __ in batch_size_dict_sample.items()]

            # for batch_size in batch_sizes:
            #     for num_eval in num_evals:
            batch_size = batch_sizes[-1]
            num_eval = num_evals[-1]

            indexes = [batch_size, num_eval]
            index_names = ["Batch Size", "Num Eval"]
            make_violin_plot(K_dict, metric_names, num_factors, val_per_factor, alphas, noise_mode, indexes,
                             index_names)
            rows_data, pd_indexes, means, stds = get_table_meanstd_info(K_dict, metric_names, num_factors,
                                                                        val_per_factor, alphas, indexes, index_names)

            # All functions append the mean-std table.
            for K in Ks:
                df = pd.DataFrame(data=rows_data[K], index=pd_indexes[K], columns=alphas)
                tableDFs[K] = tableDFs[K].append(df)

                for mean, name in zip(means[K], pd_indexes[K]):
                    all_means[K].append(mean)
                    all_names[K].append(name)
                    pass

        else:
            make_violin_plot(K_dict, metric_names, num_factors, val_per_factor, alphas, noise_mode)
            rows_data, pd_indexes, means, all_stds = get_table_meanstd_info(K_dict, metric_names, num_factors,
                                                                            val_per_factor, alphas)

            # All functions append the mean-std table.
            for K in Ks:
                df = pd.DataFrame(data=rows_data[K], index=pd_indexes[K], columns=alphas)
                tableDFs[K] = tableDFs[K].append(df)

                for mean, name in zip(means[K], pd_indexes[K]):
                    all_means[K].append(mean)
                    all_names[K].append(name)

    # Save table and fig for each K
    for K in Ks:
        tableDFs[K].to_csv("figs/{}/big_table_K_{}.csv".format(str(noise_mode), K))
        all_means[K] = np.vstack(all_means[K])
        all_names[K] = np.vstack(all_names[K])
        plot_mean_lines(all_means[K], all_names[K], alphas, K, noise_mode)

    pass
