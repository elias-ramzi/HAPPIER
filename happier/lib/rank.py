def rank(tens):
    return 1 + tens.argsort(1, True).argsort(1)
