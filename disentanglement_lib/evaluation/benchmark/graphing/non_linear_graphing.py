import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from disentanglement_lib.evaluation.benchmark.benchmark_utils import init_dict, add_to_dict, get_dict_element, get_names
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics
from matplotlib.figure import Figure

def get_table_meanstd_info(dict_,
                           metric_names,
                           num_factors,
                           val_per_factor,
                           orig_indexes=[],
                           indexes_name=[]):

    pd_indexes = []
    rows_data = []
    all_means = []
    all_stds = []

    for m_name in metric_names:

        indexes = orig_indexes.copy()
        indexes.append(m_name)

        all_scores = get_dict_element(dict_, indexes.copy())

        mean = np.mean(all_scores)
        all_means.append(mean)

        std = np.std(all_scores)
        all_stds.append(std)

        mean_std_strs = ["{:.2f} +/- {:.2f}".format(mean, std)]

        index_str = ""
        for index, index_name in zip(indexes[:-1], indexes_name):
            index_str = index_str + "{} = {} ".format(index_name, index)

        pd_indexes.append("{} {}".format(m_name, index_str))
        rows_data.append(mean_std_strs)
        pass

    return rows_data, pd_indexes, all_means, all_stds


def make_violin_plot(violin_data, labels, dict_, metric_names, num_factors, val_per_factor, nonlinear_mode, orig_indexes=[],
                     index_names=[], final_metric=False):
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
            "Non-Linear All Metrics : {}; {} Factors / {} values each".format(str(nonlinear_mode), num_factors, val_per_factor))
        plt.ylabel("Metric Score")

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 1))
        plt.legend(*zip(*legend_labels), loc='center left', bbox_to_anchor=(1, 0.5))
        x_ticks = ["({})".format(n) for n in range(0, i)]
        plt.xticks(range(1, i+1), x_ticks)
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 1.01))
        plt.savefig('figs/{}/{}_{}'.format(str(nonlinear_mode), m_name, "all_metrics_violin"), bbox_inches='tight')
        plt.close()

    return violin_data, labels


def make_graphs(results_dict, num_factors, val_per_factor, nonlinear_mode):
    if not os.path.exists('./figs/{}/'.format(str(nonlinear_mode))):
        os.mkdir('./figs/{}/'.format(str(nonlinear_mode)))

    table_df = pd.DataFrame(data=None)
    legend_labels = []
    legend_labels_extra = []
    violin_data = []
    violin_data_extra = []
    means = []
    names = []
    i = -1

    # iterate through metrics
    for f_key, dict_ in results_dict.items():
        i = i+1

        metric_names = get_names(f_key)

        # These metrics have extra common parameters (batch-size, num_eval_train)
        if f_key == Metrics.BVAE or f_key == Metrics.FVAE or f_key == Metrics.RFVAE:
            batch_size_dict_sample = results_dict[f_key]
            num_eval_dict_sample = list(batch_size_dict_sample.values())[0]

            num_evals = [num_eval for num_eval, __ in num_eval_dict_sample.items()]
            batch_sizes = [batch_size for batch_size, __ in batch_size_dict_sample.items()]

            # for batch_size in batch_sizes:
            #     for num_eval in num_evals:
            batch_size = batch_sizes[-1]
            num_eval = num_evals[-1]

            indexes = [batch_size, num_eval]
            index_names = ["Batch Size", "Num Eval"]
            fig, legend_labels = make_violin_plot(violin_data=violin_data,
                                                  labels=legend_labels,
                                                  dict_=dict_,
                                                  metric_names=metric_names,
                                                  num_factors=num_factors,
                                                  val_per_factor=val_per_factor,
                                                  nonlinear_mode=nonlinear_mode,
                                                  orig_indexes=indexes,
                                                  index_names=index_names,
                                                  final_metric=(not i < len(results_dict)-1))  # Save graph indicator

            row_data, pd_index, mean, std = get_table_meanstd_info(dict_, metric_names, num_factors,
                                                                   val_per_factor, indexes, index_names)

            # All functions append the mean-std table.
            df = pd.DataFrame(data=row_data, index=pd_index)
            table_df = table_df.append(df)

        else:
            if (f_key == Metrics.DCIMIG or f_key == Metrics.IRS or f_key == Metrics.JEMMIG or
                f_key == Metrics.MIG or f_key == Metrics.MIG_SUP or f_key == Metrics.MODEX or
                f_key == Metrics.WDG):

                # normal plotting
                make_violin_plot(violin_data=violin_data, labels=legend_labels, dict_=dict_, metric_names=metric_names, num_factors=num_factors,
                                 val_per_factor=val_per_factor, nonlinear_mode=nonlinear_mode, orig_indexes=[], index_names=[],
                                 final_metric=(not i < len(results_dict) - 1))  # Save graph indicator

                #Make special percentile discretization graph.
                make_violin_plot(violin_data=violin_data_extra, labels=legend_labels_extra, dict_=dict_, metric_names=metric_names, num_factors=num_factors,
                                 val_per_factor=val_per_factor, nonlinear_mode=nonlinear_mode, orig_indexes=[], index_names=[],
                                 final_metric=(not i < len(results_dict) - 1))  # Save graph indicator
            else:
                # normal plotting
                make_violin_plot(violin_data=violin_data, labels=legend_labels, dict_=dict_, metric_names=metric_names,
                                 num_factors=num_factors,
                                 val_per_factor=val_per_factor, nonlinear_mode=nonlinear_mode, orig_indexes=[],
                                 index_names=[],
                                 final_metric=(not i < len(results_dict) - 1))  # Save graph indicator

    table_df.to_csv("figs/{}/big_table.csv".format(str(nonlinear_mode)))
    pass
