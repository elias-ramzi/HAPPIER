def safe_mean(tens):
    if tens.nelement() != 0:
        return tens.mean()
    return tens.detach()
