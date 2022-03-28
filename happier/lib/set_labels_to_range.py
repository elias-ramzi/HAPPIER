import numpy as np


def set_labels_to_range(labels):
    """
    set the labels so it follows a range per level of semantic

    usefull for example for CSLLoss
    """
    new_labels = []
    for lvl in range(labels.shape[1]):
        unique = sorted(set(labels[:, lvl]))
        conversion = {x: i for i, x in enumerate(unique)}
        new_lvl_labels = [conversion[x] for x in labels[:, lvl]]
        new_labels.append(new_lvl_labels)

    return np.stack(new_labels, axis=1)
