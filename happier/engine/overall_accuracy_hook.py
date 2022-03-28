import numpy as np

import happier.lib as lib
from happier.engine.metrics import METRICS_DICT


def weighted_mean_hook(metrics, weights=None, name='AP'):
    to_avg = {}
    for k, v in metrics.items():
        if k.lower()[:-1] == f"{name}_level".lower():
            to_avg[int(k[-1])] = v

    if weights is None:
        weights = [1] * len(to_avg)
    elif isinstance(weights, str) and weights.startswith("geo"):
        factor = float(weights.split("_")[1])
        weights = lib.geometric_weights(len(to_avg), factor, to_tensor=False)[1:][::-1]
    elif isinstance(weights, str) and weights.startswith("halving"):
        factor = float(weights.split("_")[1])
        weights = lib.halving_weights(len(to_avg), factor, to_tensor=False)[1:][::-1]

    to_avg = [x[1] for x in sorted(to_avg.items(), key=lambda x: x[0])]
    assert len(to_avg) == len(weights)

    mean = sum([x * y for x, y in zip(to_avg, weights)]) / sum(weights)
    return mean


def inter_set_overall_accuracy_hook(metrics, look_up=METRICS_DICT):
    """
    Usefull for DyMLAnimal and DyMLVehicle.
    Calculates the overall accuracy between each level (test_fine, test_middle and test_coarse)
    """
    to_avg = list(METRICS_DICT["binary"].keys())

    overall_accuracies = {}
    for key, mtrc in metrics.items():
        if not key.startswith("test"):
            continue

        for metric_name, value in mtrc.items():
            if 'multi' in metric_name:
                continue
            if not metric_name.split("_")[0] in to_avg:
                continue

            try:
                overall_accuracies[metric_name].append(value)
            except KeyError:
                overall_accuracies[metric_name] = [value]

    for k, v in overall_accuracies.items():
        overall_accuracies[k] = np.mean(v)

    return overall_accuracies


def intra_set_overall_accuracy_hook(metrics, look_up=METRICS_DICT):
    """
    Usefull for DyMLProduct or other datasets.
    It computes the mean acc between the different level of semantic scale
    """
    to_avg = list(METRICS_DICT["binary"].keys())

    mtrc = metrics["test"].copy()
    overall_accuracies = {}
    for metric_name, value in mtrc.items():
        if not metric_name.split("_")[0] in to_avg:
            continue

        try:
            overall_accuracies[metric_name.split('_')[0]].append(value)
        except KeyError:
            overall_accuracies[metric_name.split('_')[0]] = [value]

    for k, v in overall_accuracies.items():
        overall_accuracies[k] = np.mean(v)

    return overall_accuracies


def overall_accuracy_hook(metrics, look_up=METRICS_DICT):
    names = metrics.keys()

    if sum([x.startswith("test") for x in names]) > 1:
        return inter_set_overall_accuracy_hook(metrics, look_up=look_up)
    else:
        return intra_set_overall_accuracy_hook(metrics, look_up=look_up)
