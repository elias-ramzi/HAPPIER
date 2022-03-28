import sys
from os.path import join

import torch

import happier.lib as lib


def checkpoint(
    log_dir,
    save_checkpoint,
    net,
    optimizer,
    scheduler,
    criterion,
    scaler,
    epoch,
    config,
    metrics,
):
    state_dict = {}
    if torch.cuda.device_count() > 1:
        state_dict["net_state"] = net.module.state_dict()
    else:
        state_dict["net_state"] = net.state_dict()

    state_dict["optimizer_state"] = {key: opt.state_dict() for key, opt in optimizer.items()}

    state_dict["scheduler_on_epoch_state"] = [sch.state_dict() for sch in scheduler["on_epoch"]]
    state_dict["scheduler_on_step_state"] = [sch.state_dict() for sch in scheduler["on_step"]]
    state_dict["scheduler_on_val_state"] = [sch.state_dict() for sch, _ in scheduler["on_val"]]

    state_dict["criterion_state"] = [crit.state_dict() for crit, _ in criterion]

    if scaler is not None:
        state_dict["scaler_state"] = scaler.state_dict()

    state_dict["epoch"] = epoch
    state_dict["config"] = config
    state_dict["command"] = 'python ' + ' '.join(sys.argv)
    state_dict["metrics"] = metrics

    RANDOM_STATE = lib.get_random_state()
    state_dict.update(RANDOM_STATE)

    torch.save(state_dict, join(log_dir, 'weights', "rolling.ckpt"))
    if save_checkpoint:
        lib.LOGGER.info(f"Checkpoint of epoch {epoch} created")
        torch.save(state_dict, join(log_dir, 'weights', f"epoch_{epoch}.ckpt"))
