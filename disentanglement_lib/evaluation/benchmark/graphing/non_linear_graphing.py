import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from disentanglement_lib.evaluation.benchmark.benchmark_utils import init_dict, add_to_dict, get_dict_element, get_names
from disentanglement_lib.config.benchmark.scenarios.bindings import Metrics
from matplotlib.figure import Figure

def get_list_dict_from_ids(_list_to_sample, ids):
    _list = np.take(_list_to_sample, ids, axis=0).flatten()
    _dict = group_list_in_dict(_list)
    return _list, _dict

def group_list_in_dict(results_dict_list):
    grouped_dict = {}
    grouped_dict["extra_params"] = [dict_["extra_params"] for dict_ in results_dict_list]
    grouped_dict["scores"] = [dict_["score"] for dict_ in results_dict_list]

    grouped_dict["param_names"] = results_dict_list[0]["param_names"]
    grouped_dict["extra_params_unique"] = np.unique(grouped_dict["extra_params"], axis=0)

    grouped_dict["seeds"] = [dict_["seed"] for dict_ in results_dict_list]
    grouped_dict["n_seeds"] = np.max(grouped_dict["seeds"]) + 1
    return grouped_dict


def make_violin_plot(violin_data, labels, fn_results_list, metric_names, num_factors, val_per_factor, nonlinear_mode, final_metric=False, mode="all"):
    def add_label(violin, label, violin_color):
        legend_labels.append((mpatches.Patch(color=violin_color), label))

    grouped_dict = group_list_in_dict(fn_results_list)
    metric_data = []
    metric_labels = []
    for m_name in metric_names:
        for extra_params in grouped_dict["extra_params_unique"]:
            param_list, param_dict = get_list_dict_from_ids(fn_results_list,
                                                            np.argwhere((grouped_dict["extra_params"] == extra_params).all(-1)))
            scores = [dict_["score"][m_name] for dict_ in param_list]
            legend_str = ""
            for index, index_name in zip(extra_params, grouped_dict["param_names"]):
                legend_str = legend_str + "{}_{} ".format(index_name, index)
            label = "{}_{}".format(m_name, legend_str)
            metric_data.append(scores)
            metric_labels.append(label)
    #one element per metric.
    violin_data.append(metric_data)
    labels.append(metric_labels)

    if final_metric:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        legend_labels = []
        i = 0
        figure = plt.gcf()
        figure.set_size_inches(30, 13)

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
        plt.savefig('figs/{}/{}_{}'.format(str(nonlinear_mode), "metrics_violin", mode), bbox_inches='tight')
        plt.close()

    return violin_data, labels



def make_graphs(results_dict_list, num_factors, val_per_factor, nonlinear_mode):
    if not os.path.exists('./figs/{}/'.format(str(nonlinear_mode))):
        os.mkdir('./figs/{}/'.format(str(nonlinear_mode)))

    legend_labels = []
    violin_data = []
    i = -1
    # iterate through metrics
    for f_key, fn_results_list in results_dict_list.items():
        i = i+1
        metric_names = get_names(f_key)
        violin_data, violin_labels = make_violin_plot(violin_data, legend_labels,
                                                      fn_results_list, metric_names, num_factors, val_per_factor, nonlinear_mode,
                                                      final_metric=(not i < len(list(results_dict_list.keys())) - 1))  # Save graph indicator

    legend_labels = []
    violin_data = []
    i = -1
    for f_key, fn_results_list in results_dict_list.items():
        i = i+1
        metric_names = get_names(f_key, mode="parsed")
        violin_data, violin_labels = make_violin_plot(violin_data, legend_labels,
                                                      fn_results_list, metric_names, num_factors, val_per_factor, nonlinear_mode,
                                                      final_metric=(not i < len(list(results_dict_list.keys())) - 1), mode="parsed")  # Save graph indicator
