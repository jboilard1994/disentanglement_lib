import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from disentanglement_lib.evaluation.benchmark.benchmark_utils import init_dict, add_to_dict, get_dict_element, get_names
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics
from disentanglement_lib.evaluation.benchmark.scenarios.modcompact_dataholder import ModCompactMode


def get_table_meanstd_info(_dict,
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

        all_scores = np.array([get_dict_element(d, indexes.copy()) for alpha_val, d in _dict.items()])

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


def make_violin_plot(alpha_dict, metric_names, num_factors, val_per_factor, alphas, mod_compact_mode, orig_indexes=[],
                     indexes_name=[]):
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    for m_name in metric_names:
        indexes = orig_indexes.copy()
        indexes.append(m_name)

        labels = []
        all_scores = [get_dict_element(d, indexes.copy()) for alpha_val, d in alpha_dict.items()]
        plt.violinplot(all_scores, showmeans=True)

        title_str = ""
        for index, index_name in zip(orig_indexes, indexes_name):
            title_str = title_str + "{}_{} ".format(index_name, index)

        plt.title(
            "{}: {}".format(m_name, str(mod_compact_mode)) + title_str + "{} Factors / {} values each".format(num_factors,
                                                                                                              val_per_factor))
        plt.xticks(range(1, len(alphas) + 1), alphas)
        plt.xlabel("Noise-signal ratio")
        plt.ylabel(m_name)
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 1.01))
        plt.savefig('figs/{}/{}_{}'.format(str(mod_compact_mode), m_name, title_str), bbox_inches='tight')
        plt.close()


def plot_mean_lines(means, mnames, alphas, noise_mode):
    for mean, mname in zip(means, mnames):
        plt.plot(alphas, mean, label=mname)
    plt.title("Mean values of metrics according to alpha")
    plt.xlabel("alpha")
    plt.ylabel("Metric Results")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('figs/{}/metric_means.png'.format(str(noise_mode)), bbox_inches='tight')
    plt.close()
    pass

def make_graphs(results_dict, num_factors, val_per_factor, mod_compact_mode):
    if not os.path.exists('./figs/{}/'.format(str(mod_compact_mode))):
        os.mkdir('./figs/{}/'.format(str(mod_compact_mode)))

    if mod_compact_mode == ModCompactMode.TEST_MOD_REDUCE:
        num_factors = num_factors*2

    if mod_compact_mode == ModCompactMode.TEST_MOD_MISSING_CHECK or mod_compact_mode == ModCompactMode.TEST_COMPACT_MISSING_CHECK:
        make_graphs_no_alpha(results_dict, num_factors, val_per_factor, mod_compact_mode)
    else:
        # Define some necessary values found in dict.
        alpha_dict_sample = list(results_dict.values())[0]

        alphas = [alpha_val for alpha_val, __ in alpha_dict_sample.items()]

        all_means = []
        all_names = []
        tableDF = pd.DataFrame(data=None, columns=alphas)

        # iterate through metrics
        for f_key, alpha_dict in results_dict.items():

            metric_names = get_names(f_key)

            # These metrics have extra common parameters (batch-size, num_eval_train)
            if f_key == Metrics.BVAE or f_key == Metrics.FVAE or f_key == Metrics.RFVAE:
                batch_size_dict_sample = list(alpha_dict.values())[0]
                num_eval_dict_sample = list(batch_size_dict_sample.values())[0]
                num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
                batch_sizes = [batch_size for batch_size, __ in batch_size_dict_sample.items()]

                # for batch_size in batch_sizes:
                #     for num_eval in num_evals:
                batch_size = batch_sizes[-1]
                num_eval = num_evals[-1]

                indexes = [batch_size, num_eval]
                index_names = ["Batch Size", "Num Eval"]
                make_violin_plot(alpha_dict, metric_names, num_factors, val_per_factor, alphas, mod_compact_mode, indexes,
                                 index_names)
                rows_data, pd_indexes, means, stds = get_table_meanstd_info(alpha_dict, metric_names, num_factors,
                                                                            val_per_factor, alphas, indexes, index_names)

                # All functions append the mean-std table.
                df = pd.DataFrame(data=rows_data, index=pd_indexes, columns=alphas)
                tableDF = tableDF.append(df)

                for mean, name in zip(means, pd_indexes):
                    all_means.append(mean)
                    all_names.append(name)
                    pass

            else:
                make_violin_plot(alpha_dict, metric_names, num_factors, val_per_factor, alphas, mod_compact_mode)
                rows_data, pd_indexes, means, all_stds = get_table_meanstd_info(alpha_dict, metric_names, num_factors,
                                                                                val_per_factor, alphas)

                # All functions append the mean-std table.
                df = pd.DataFrame(data=rows_data, index=pd_indexes, columns=alphas)
                tableDF = tableDF.append(df)

                for mean, name in zip(means, pd_indexes):
                    all_means.append(mean)
                    all_names.append(name)

        # Save table and fig for each K
        tableDF.to_csv("figs/{}/big_table.csv".format(str(mod_compact_mode)))
        all_means = np.vstack(all_means)
        all_names = np.vstack(all_names)
        plot_mean_lines(all_means, all_names, alphas, mod_compact_mode)
        pass


def make_violin_plot_no_alpha(violin_data, labels, dict_, metric_names, num_factors, val_per_factor, modcompact_mode,
                                                          orig_indexes=[], index_names=[], final_metric=False):
    def add_label(violin, label, violin_color):
        legend_labels.append((mpatches.Patch(color=violin_color), label))

    metric_data = []
    metric_labels = []
    for m_name in metric_names:
        indexes = orig_indexes.copy()
        indexes.append(m_name)

        legend_str = ""
        for index, index_name in zip(orig_indexes, index_names):
            legend_str = legend_str + "{}_{} ".format(index_name, index)
        seed_scores = get_dict_element(dict_, indexes.copy())
        label = "{}_{}".format(m_name, legend_str)

        metric_data.append(seed_scores)
        metric_labels.append(label)
    violin_data.append(metric_data)
    labels.append(metric_labels)

    if final_metric:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        legend_labels = []
        i = 0
        n_scores = len(seed_scores)
        figure = plt.gcf()
        figure.set_size_inches(24, 13)

        for metric_scores, metric_labels in zip(violin_data, labels):
            for n in range(len(metric_scores)):
                p = plt.violinplot(metric_scores[n], [i + 1], showmeans=True)

                # Set common color for all sub-metrics.
                if n == 0:
                    face_color = colors[i % len(colors)]
                    edge_color = colors[i % len(colors)]

                p['bodies'][0].set_facecolor(face_color)
                for part_name in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                    p[part_name].set_edgecolor(edge_color)

                add_label(p, "({}) {}".format(i, metric_labels[n]), face_color)
                i = i+1


        plt.title(
            "Non-Linear All Metrics : {}; {} Factors / {} values each".format(str(modcompact_mode), num_factors, val_per_factor))
        plt.ylabel("Metric Score")

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 1))
        plt.legend(*zip(*legend_labels), loc='center left', bbox_to_anchor=(1, 0.5))
        x_ticks = ["({})".format(n) for n in range(0, i)]
        plt.xticks(range(1, i+1), x_ticks)
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 1.01))
        plt.savefig('figs/{}/{}_{}'.format(str(modcompact_mode), m_name, "all_metrics_violin"), bbox_inches='tight')
        plt.close()

    return violin_data, labels


def make_graphs_no_alpha(results_dict, num_factors, val_per_factor, modcompact_mode):
    if not os.path.exists('./figs/{}/'.format(str(modcompact_mode))):
        os.mkdir('./figs/{}/'.format(str(modcompact_mode)))

    legend_labels = []
    violin_data = []
    means = []
    names = []
    i = -1
    alpha = 0

    # iterate through metrics
    for f_key, dict_ in results_dict.items():
        i = i+1

        metric_names = get_names(f_key)

        # These metrics have extra common parameters (batch-size, num_eval_train)
        if f_key == Metrics.BVAE or f_key == Metrics.FVAE or f_key == Metrics.RFVAE:
            batch_size_dict_sample = results_dict[f_key][alpha]
            num_eval_dict_sample = list(batch_size_dict_sample.values())[0]

            num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
            batch_sizes = [batch_size for batch_size, __ in batch_size_dict_sample.items()]

            # for batch_size in batch_sizes:
            #     for num_eval in num_evals:
            batch_size = batch_sizes[-1]
            num_eval = num_evals[-1]

            indexes = [alpha, batch_size, num_eval]
            index_names = ["", "Batch Size", "Num Eval"]
            fig, legend_labels = make_violin_plot_no_alpha(violin_data=violin_data,
                                                           labels=legend_labels,
                                                           dict_=dict_,
                                                           metric_names=metric_names,
                                                           num_factors=num_factors,
                                                           val_per_factor=val_per_factor,
                                                           modcompact_mode=modcompact_mode,
                                                           orig_indexes=indexes,
                                                           index_names=index_names,
                                                           final_metric=(not i < len(results_dict)-1))  # Save graph indicator

        else:
            fig, legend_labels = make_violin_plot_no_alpha(violin_data=violin_data,
                                                           labels=legend_labels,
                                                           dict_=dict_,
                                                           metric_names=metric_names,
                                                           num_factors=num_factors,
                                                           val_per_factor=val_per_factor,
                                                           modcompact_mode=modcompact_mode,
                                                           orig_indexes=[alpha],
                                                           index_names=[""],
                                                           final_metric=(not i < len(results_dict) - 1))  # Save graph indicator
    pass
