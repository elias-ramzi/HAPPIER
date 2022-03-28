import os
import logging
import argparse

import torch
import numpy as np
from omegaconf import open_dict

import happier.lib as lib
import happier.engine as eng
from happier.getter import Getter


def print_metrics(metrics):
    for split, mtrc in metrics.items():
        for k, v in mtrc.items():
            if k == 'epoch':
                continue
            lib.LOGGER.info(f"{split} --> {k} : {np.around(v*100, decimals=2)}")
        if split in ['test']:
            lib.LOGGER.info(f"This is for latex--> {lib.format_for_latex(mtrc)}")
        if split in ['test_overall']:
            lib.LOGGER.info(f"This is for DyLM latex--> {lib.format_for_latex_dyml(mtrc)}")
        print()
        print()


def load_and_evaluate(
    path,
    hierarchy_level,
    set,
    relevance_type,
    factor,
    ibs,
    mbs,
    nw,
    data_dir=None,
    no_amp=False,
    inat_base=False,
    inat_full=False,
):
    os.environ['USE_CUDA_FOR_RELEVANCE'] = 'yes'
    lib.LOGGER.info(f"Evaluating : \033[92m{path}\033[0m")
    state = torch.load(lib.expand_path(path), map_location='cpu')
    cfg = state["config"]

    # TEMP:
    cfg.experience.accuracy_calculator.exclude = []
    with open_dict(cfg.experience.accuracy_calculator):
        cfg.experience.accuracy_calculator.with_binary_asi = True

    if factor:
        cfg.dataset.kwargs.factor = factor[0]
    if relevance_type:
        cfg.dataset.kwargs.relevance_type = relevance_type

    lib.LOGGER.info("Loading model...")
    state["config"].model.kwargs.with_autocast = not no_amp
    net = Getter().get_model(state["config"].model)
    net.load_state_dict(state["net_state"])
    if torch.cuda.device_count() > 1:
        lib.LOGGER.info("Model running on multiple GPU's...")
        net = torch.nn.DataParallel(net)
    net.cuda()
    net.eval()

    if data_dir is not None:
        cfg.dataset.kwargs.data_dir = lib.expand_path(data_dir)

    if inat_base:
        assert not inat_full
        if hasattr(cfg.dataset.kwargs, 'hierarchy_mode'):
            cfg.dataset.kwargs.hierarchy_mode = 'base'
        else:
            with open_dict(cfg.dataset.kwargs):
                cfg.dataset.kwargs.hierarchy_mode = 'base'
        hierarchy_level = [0, 1]

    if inat_full:
        assert not inat_base
        if hasattr(cfg.dataset.kwargs, 'hierarchy_mode'):
            cfg.dataset.kwargs.hierarchy_mode = 'full'
        else:
            with open_dict(cfg.dataset.kwargs):
                cfg.dataset.kwargs.hierarchy_mode = 'full'
        hierarchy_level = [0, 1, 2, 3, 4, 5, 6]

    getter = Getter()
    transform = getter.get_transform(cfg.transform.test)
    dts = getter.get_dataset(transform, set, cfg.dataset)

    if set == 'test':
        dataset_dict = {}
        if isinstance(dts, list):
            for i, _dts in enumerate(dts):
                dataset_dict[f"test_level{i}"] = _dts
        else:
            dataset_dict["test"] = dts
    else:
        dataset_dict = {set: dts}

    lib.LOGGER.info("Dataset created...")

    cfg.experience.accuracy_calculator.num_workers = nw
    cfg.experience.accuracy_calculator.inference_batch_size = ibs
    cfg.experience.accuracy_calculator.metric_batch_size = mbs
    if hierarchy_level is not None:
        cfg.experience.accuracy_calculator.compute_for_hierarchy_levels = hierarchy_level
    acc = getter.get_acc_calculator(cfg.experience)

    metrics = eng.evaluate(
        net=net,
        dataset_dict=dataset_dict,
        acc=acc,
        epoch=state["epoch"],
    )

    lib.LOGGER.info("Evaluation completed...")
    print_metrics(metrics)

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Path to checkpoint')
    parser.add_argument("--hierarchy-level", type=int, default=None, nargs='+', help='Hierarchy level for Acc')
    parser.add_argument("--set", type=str, default='test', help='Set on which to evaluate')
    parser.add_argument("--relevance-type", type=str, default=None, help='Relevance type')
    parser.add_argument("--factor", type=float, nargs='+', default=None, help='Factor to compute the H-AP and NDCG')
    parser.add_argument("--iBS", type=int, default=256, help='Batch size for DataLoader')
    parser.add_argument("--mBS", type=int, default=256, help='Batch size for metric calculation')
    parser.add_argument("--nw", type=int, default=10, help='Num workers for DataLoader')
    parser.add_argument("--data-dir", type=str, default=None, help='Possible override of the datadir in the dataset config')
    parser.add_argument("--no-amp", default=True, action='store_false', help='Deactivates mix precision')
    parser.add_argument("--metrics-from-checkpoint", default=False, action='store_true', help='Only reads the metrics in the checkpoint')
    parser.add_argument("--inat-base", default=False, action='store_true', help='Allow the use of a model trained on INat-Full to be evaluated on INat-Base')
    parser.add_argument("--inat-full", default=False, action='store_true', help='Allow the use of a model trained on INat-Full to be evaluated on INat-Base')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    if args.metrics_from_checkpoint:
        metrics = torch.load(args.config, map_location='cpu')['metrics']
        lib.LOGGER.info(f"READING : \033[92m{args.config}\033[0m")
        print_metrics(metrics)
    else:
        metrics = load_and_evaluate(
            path=args.config,
            hierarchy_level=args.hierarchy_level,
            set=args.set,
            relevance_type=args.relevance_type,
            factor=args.factor,
            ibs=args.iBS,
            mbs=args.mBS,
            nw=args.nw,
            data_dir=args.data_dir,
            no_amp=args.no_amp,
            inat_base=args.inat_base,
            inat_full=args.inat_full,
        )
        print()
        print()
