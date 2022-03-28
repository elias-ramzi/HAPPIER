import os
from os.path import join
import random

import numpy as np
from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import happier.lib as lib
import happier.engine as eng
from happier.getter import Getter


def if_func(cond, x, y):
    if not isinstance(cond, bool):
        cond = eval(cond)
        assert isinstance(cond, bool)
    if cond:
        return x
    return y


OmegaConf.register_new_resolver("mult", lambda *numbers: np.prod([float(x) for x in numbers]))
OmegaConf.register_new_resolver("sum", lambda *numbers: sum(map(float, numbers)))
OmegaConf.register_new_resolver("sub", lambda x, y: float(x) - float(y))
OmegaConf.register_new_resolver("div", lambda x, y: float(x) / float(y))
OmegaConf.register_new_resolver("if", if_func)


@hydra.main(config_path='config', config_name='default')
def run(config):
    """
    creates all objects required to launch a training
    """
    # """""""""""""""""" Handle Config """"""""""""""""""""""""""
    config.experience.log_dir = lib.expand_path(config.experience.log_dir)
    log_dir = join(config.experience.log_dir, config.experience.experiment_name)

    if 'debug' in config.experience.experiment_name.lower():
        config.experience.DEBUG = config.experience.DEBUG or 1

    if config.experience.resume is not None:
        if os.path.isfile(lib.expand_path(config.experience.resume)):
            resume = lib.expand_path(config.experience.resume)
        else:
            resume = os.path.join(log_dir, 'weights', config.experience.resume)
            if not os.path.isfile(resume):
                lib.LOGGER.warning("Checkpoint does not exists")
                return

        state = torch.load(resume, map_location='cpu')
        at_epoch = state["epoch"]
        if at_epoch >= config.experience.max_iter:
            lib.LOGGER.warning(f"Exiting trial, experiment {config.experience.experiment_name} already finished")
            return

        lib.LOGGER.info(f"Resuming from state : {resume}")
        restore_epoch = state['epoch']

    else:
        resume = None
        state = None
        restore_epoch = 0
        if os.path.isdir(os.path.join(log_dir, 'weights')) and not config.experience.DEBUG:
            lib.LOGGER.warning(f"Exiting trial, experiment {config.experience.experiment_name} already exists")
            lib.LOGGER.warning(f"Its access: {log_dir}")
            return

    os.makedirs(join(log_dir, 'logs'), exist_ok=True)
    os.makedirs(join(log_dir, 'weights'), exist_ok=True)
    writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    # """""""""""""""""" Handle Reproducibility"""""""""""""""""""""""""
    lib.LOGGER.info(f"Training with seed {config.experience.seed}")
    random.seed(config.experience.seed)
    np.random.seed(config.experience.seed)
    torch.manual_seed(config.experience.seed)
    torch.cuda.manual_seed_all(config.experience.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    os.environ['USE_CUDA_FOR_RELEVANCE'] = 'yes'
    getter = Getter()

    train_transform = getter.get_transform(config.transform.train)
    test_transform = getter.get_transform(config.transform.test)
    train_dts = getter.get_dataset(train_transform, 'train', config.dataset)
    test_dts = getter.get_dataset(test_transform, 'test', config.dataset)
    val_dts = None

    sampler = getter.get_sampler(train_dts, config.dataset.sampler)

    # """""""""""""""""" Create Network """"""""""""""""""""""""""
    net = getter.get_model(config.model)

    scaler = None
    if config.model.kwargs.with_autocast:
        scaler = torch.cuda.amp.GradScaler()
        if state is not None:
            scaler.load_state_dict(state['scaler_state'])

    if state is not None:
        net.load_state_dict(state['net_state'])
        net.cuda()

    # """""""""""""""""" Create Optimizer & Scheduler """"""""""""""""""""""""""
    optimizer, scheduler = getter.get_optimizer(net, config.optimizer)

    if state is not None:
        for key, opt in optimizer.items():
            opt.load_state_dict(state['optimizer_state'][key])

        for key, sch in scheduler.items():
            sch.load_state_dict(state[f'scheduler_{key}_state'])

    # """""""""""""""""" Create Criterion """"""""""""""""""""""""""
    criterion = getter.get_loss(config.loss)

    for crit, _ in criterion:
        if hasattr(crit, 'register_labels'):
            crit.register_labels(torch.from_numpy(train_dts.labels))

    if state is not None and "criterion_state" in state:
        for (crit, _), crit_state in zip(criterion, state["criterion_state"]):
            crit.cuda()
            crit.load_state_dict(crit_state)

    acc = getter.get_acc_calculator(config.experience)

    # """""""""""""""""" Handle Cuda """"""""""""""""""""""""""
    if torch.cuda.device_count() > 1:
        lib.LOGGER.info("Model is parallelized")
        net = nn.DataParallel(net)

    if config.experience.parallelize_loss:
        for i, (crit, w) in enumerate(criterion):
            level = crit.hierarchy_level
            crit = nn.DataParallel(crit)
            crit.hierarchy_level = level

    net.cuda()
    _ = [crit.cuda() for crit, _ in criterion]

    # """""""""""""""""" Handle RANDOM_STATE """"""""""""""""""""""""""
    if state is not None:
        # set random NumPy and Torch random states
        lib.set_random_state(state)

    return eng.train(
        config=config,
        log_dir=log_dir,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        acc=acc,
        train_dts=train_dts,
        val_dts=val_dts,
        test_dts=test_dts,
        sampler=sampler,
        writer=writer,
        restore_epoch=restore_epoch,
    )


if __name__ == '__main__':
    run()
