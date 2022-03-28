from collections import OrderedDict


def adapt_checkpoint(state_dict, remove='module.'):
    new_dict = OrderedDict()
    for key, weight in state_dict.items():
        new_key = key.replace(remove, "")
        new_dict[new_key] = weight
    return new_dict
