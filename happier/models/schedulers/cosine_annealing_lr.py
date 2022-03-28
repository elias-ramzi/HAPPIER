from torch.optim import lr_scheduler


class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):

    def __repr__(self):
        repr = f"{self.__class__.__name__}(\n"
        repr = repr + f"    T_max={self.T_max},\n"
        repr = repr + f"    eta_min={self.eta_min},\n"
        repr = repr + ')'
        return repr
