def identity(x):
    return x


def cast_dict_to_type(dct, keys_type=None, values_type=None):
    assert isinstance(dct, dict)
    assert (keys_type is not None) or (values_type is not None)

    if keys_type is None:
        keys_type = identity

    if values_type is None:
        values_type = identity

    return {keys_type(k): values_type(v) for k, v in dct.items()}
