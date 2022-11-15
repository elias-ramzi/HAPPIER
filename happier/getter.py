import torch
from torch import optim
import torchvision.transforms as T

import happier.lib as lib
from happier import losses
from happier import datasets
from happier import models
from happier import engine
from happier.models import schedulers


class Getter:
    """
    This class allows to create differents object (model, loss functions, optimizer...)
    based on the config
    """
    def get_transform(self, config, num_crops=-1):
        t_list = []
        for k, v in config.items():
            if k == 'RandomApply':
                t_list.append(T.RandomApply(getattr(T, v.aug)(**v.kwargs), p=v.ps))
            else:
                t_list.append(getattr(T, k)(**v))

        transform = T.Compose(t_list)
        lib.LOGGER.info(transform)
        return transform

    def get_optimizer(self, net, config):
        optimizers = {}
        all_schedulers = {
            "on_epoch": [],
            "on_step": [],
            "on_val": [],
        }
        for opt in config['opt']:
            optimizer = getattr(optim, opt.name)
            if opt.params is not None:
                optimizer = optimizer(getattr(net, opt.params).parameters(), **opt.kwargs)
                optimizers[opt.params] = optimizer
            else:
                optimizer = optimizer(net.parameters(), **opt.kwargs)
                optimizers["net"] = optimizer
            lib.LOGGER.info(optimizer)
            if opt.scheduler_on_epoch is not None:
                all_schedulers["on_epoch"].append(self.get_scheduler(optimizer, opt.scheduler_on_epoch))
            if opt.scheduler_on_step is not None:
                all_schedulers["on_step"].append(self.get_scheduler(optimizer, opt.scheduler_on_step))
            if opt.scheduler_on_val is not None:
                all_schedulers["on_val"].append(
                    (self.get_scheduler(optimizer, opt.scheduler_on_val), opt.scheduler_on_val.key)
                )

        return optimizers, all_schedulers

    def get_scheduler(self, opt, config):
        if hasattr(schedulers, config.name):
            sch = getattr(schedulers, config.name)(opt, **config.kwargs)
        else:
            sch = getattr(optim.lr_scheduler, config.name)(opt, **config.kwargs)
        lib.LOGGER.info(sch)
        return sch

    def get_loss(self, config):
        criterion = []
        for i, crit in enumerate(config.losses):
            if hasattr(crit, 'script') and crit.script is not None:
                lib.LOGGER.info("Scripting loss")
                # https://github.com/pytorch/pytorch/issues/61382
                torch._C._jit_override_can_fuse_on_gpu(False)
                loss = torch.jit.script(getattr(losses, crit.name)(**crit.kwargs))
            else:
                loss = getattr(losses, crit.name)(**crit.kwargs)

            if hasattr(crit, 'optimizer') and crit.optimizer is not None:
                crit.optimizer.params = None
                opt, schedulers = self.get_optimizer(loss, {'opt': [crit.optimizer]})
                lib.LOGGER.info(f"Adding optimizer to {crit.name}")
                loss.register_optimizers(opt["net"], {k: v[0] if v else None for k, v in schedulers.items()})

            if hasattr(crit, 'inner_optimizer') and crit.optimizer is not None:
                crit.optimizer.params = None
                opt, schedulers = self.get_optimizer(loss, {'opt': [crit.inner_optimizer]})
                lib.LOGGER.info(f"Adding inner optimizer to {crit.name}")
                loss.register_inner_optimizers(opt["net"])

            weight = crit.weight
            if isinstance(weight, str) and weight.startswith('geo'):
                factor = float(weight.split("_")[1])
                # [1:] : geometric_weights returns a weight of 0 for the null relevance
                # [::-1] : in my config files the fined grain loss is first
                weight = lib.geometric_weights(len(config.losses), factor, to_tensor=False)[1:][::-1][i]
            elif isinstance(weight, str) and weight.startswith('eq'):
                weight = 1 / len(config.losses)
            lib.LOGGER.info(f"{loss} with weight {weight}")
            criterion.append((loss, weight))
        return criterion

    def get_sampler(self, dataset, config):
        sampler = getattr(datasets, config.name)(dataset, **config.kwargs)
        lib.LOGGER.info(sampler)
        return sampler

    def get_dataset(self, transform, mode, config):
        if (config.name == "InShopDataset") and (mode == "test"):
            dataset = {
                "query": getattr(datasets, config.name)(transform=transform, mode="query", **config.kwargs),
                "gallery": getattr(datasets, config.name)(transform=transform, mode="gallery", **config.kwargs),
            }
            lib.LOGGER.info(dataset)
            return dataset
        elif (config.name == "DyMLDataset") and mode.startswith("test"):
            lib.LOGGER.warning("Not using custom splits for DyML")
            dts_fn = getattr(datasets, config.name)
            dataset = [
                {
                    "query": dts_fn(transform=transform, mode="test_query_fine", **config.kwargs),
                    "gallery_distractor": dts_fn(transform=transform, mode="test_gallery_fine", **config.kwargs),
                },
                {
                    "query": dts_fn(transform=transform, mode="test_query_middle", **config.kwargs),
                    "gallery_distractor": dts_fn(transform=transform, mode="test_gallery_middle", **config.kwargs),
                },
                {
                    "query": dts_fn(transform=transform, mode="test_query_coarse", **config.kwargs),
                    "gallery_distractor": dts_fn(transform=transform, mode="test_gallery_coarse", **config.kwargs),
                },
            ]
            lib.LOGGER.info(dataset)
            return dataset
        else:
            dataset = getattr(datasets, config.name)(
                transform=transform,
                mode=mode,
                **config.kwargs,
            )
            lib.LOGGER.info(dataset)
            return dataset

    def get_model(self, config):
        net = getattr(models, config.name)(**config.kwargs)
        if config.freeze_batch_norm:
            lib.LOGGER.info("Freezing batch norm")
            net = lib.freeze_batch_norm(net)
        else:
            lib.LOGGER.info("/!\\ Not freezing batch norm")
        return net

    def get_memory(self, config):
        memory = getattr(engine, config.name)(**config.kwargs)
        lib.LOGGER.info(memory)
        return memory

    def get_acc_calculator(self, config):
        acc = getattr(engine, 'AccuracyCalculator')(**config.accuracy_calculator)
        lib.LOGGER.info(acc)
        return acc
