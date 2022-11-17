import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import happier.lib as lib

from happier.engine.get_knn import get_knn
from happier.engine.metrics import get_metrics_dict
from happier.engine.compute_embeddings import compute_embeddings
from happier.engine.overall_accuracy_hook import overall_accuracy_hook


class AccuracyCalculator:

    def __init__(
        self,
        compute_for_hierarchy_levels=[0],
        exclude=[],
        recall_rate=[],
        hard_ap_for_level=[],
        with_binary_asi=False,
        overall_accuracy=False,
        metric_batch_size=256,
        inference_batch_size=256,
        num_workers=16,
        pin_memory=True,
        convert_to_cuda=True,
        with_faiss=False,
        **kwargs,
    ):
        self.compute_for_hierarchy_levels = sorted(set(compute_for_hierarchy_levels))
        self.exclude = exclude
        self.recall_rate = recall_rate
        self.hard_ap_for_level = hard_ap_for_level
        self.with_binary_asi = with_binary_asi
        self.overall_accuracy = overall_accuracy
        self.metric_batch_size = metric_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.convert_to_cuda = convert_to_cuda
        self.with_faiss = with_faiss

        self.METRICS_DICT = get_metrics_dict(
            recall_rate=self.recall_rate,
            hard_ap_for_level=self.hard_ap_for_level,
            with_binary_asi=self.with_binary_asi,
            **kwargs,
        )

    def get_embeddings(self, net, dts):
        loader = DataLoader(
            dts,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return compute_embeddings(
            net,
            loader,
            self.convert_to_cuda,
        )

    def batch_metrics(
        self,
        features,
        labels,
        embeddings_come_from_same_source,
        relevances=None,
        ref_features=None,
        ref_labels=None,
    ):
        lib.LOGGER.info("Launching batched metrics")
        if ref_features is None:
            assert ref_labels is None
            ref_features, ref_labels = features, labels

        BS = self.metric_batch_size

        out = lib.DictAverage()
        iterator = tqdm(range(features.size(0) // BS + (features.size(0) % BS != 0)), disable=os.getenv("TQDM_DISABLE"))
        for i in iterator:
            logs = {}
            indices, distances = get_knn(
                ref_features, features[i*BS:(i+1)*BS],
                ref_features.size(0)-1,
                embeddings_come_from_same_source=embeddings_come_from_same_source,
                with_faiss=self.with_faiss,
            )

            in_dict = {}
            in_dict['sorted_target'] = lib.create_label_matrix(labels[i*BS:(i+1)*BS], ref_labels[indices], dtype=torch.int64)
            if relevances is not None:
                in_dict['relevances'] = relevances[i*BS:(i+1)*BS]
                in_dict['sorted_rel'] = lib.create_relevance_matrix(
                    in_dict['sorted_target'],
                    in_dict['relevances']
                )
            for key, metric in self.METRICS_DICT["multi_level"].items():
                if key not in self.exclude:
                    logs[f"{key}_multi"] = metric(**in_dict)

            for key, metric in self.METRICS_DICT["exclude_level"].items():
                if key not in self.exclude:
                    logs[f"{key}_exclude"] = metric(**in_dict)

            for hierarchy_level in self.compute_for_hierarchy_levels:
                # creates a binary label matrix corresponding to hierarchy_level
                binary_sorted_target = lib.create_label_matrix(
                    labels[i*BS:(i+1)*BS],
                    ref_labels[indices],
                    hierarchy_level,
                ).float()
                for key, metric in self.METRICS_DICT["binary"].items():
                    if key not in self.exclude:
                        logs[f"{key}_level{hierarchy_level}"] = metric(binary_sorted_target)

            out.update(logs, in_dict['sorted_target'].size(0))
            iterator.set_postfix(out.avg)

        return out.avg

    def evaluate(
        self,
        net,
        dataset_dict,
        epoch=None,
    ):
        if epoch is not None:
            lib.LOGGER.info(f"Evaluating for epoch {epoch}")

        logs = {}
        for split, dts in dataset_dict.items():
            if isinstance(dts, Dataset):
                lib.LOGGER.info(f"Getting embeddings for the {split} set")
                features, labels, relevances = self.get_embeddings(net, dts)
                ref_features = ref_labels = None
                embeddings_come_from_same_source = True

            elif isinstance(dts, dict):
                if 'gallery' in dts.keys():
                    # gallery and queries are disjoint
                    lib.LOGGER.info(f"Getting embeddings for the queries of the {split} set")
                    features, labels, relevances = self.get_embeddings(net, dts["query"])
                    lib.LOGGER.info(f"Getting embeddings for the gallery of the {split} set")
                    ref_features, ref_labels, _ = self.get_embeddings(net, dts["gallery"])
                    embeddings_come_from_same_source = False

                elif 'distractor' in dts.keys():
                    # gallery is composed of queries + distractors
                    lib.LOGGER.info(f"Getting embeddings for the queries of the {split} set")
                    features, labels, relevances = self.get_embeddings(net, dts["query"])
                    lib.LOGGER.info(f"Getting embeddings for the disctractor of the {split} set")
                    ref_features, ref_labels, _ = self.get_embeddings(net, dts["disctractor"])
                    embeddings_come_from_same_source = False
                    ref_features = torch.cat((features, ref_features), dim=0)
                    ref_labels = torch.cat((labels, ref_labels), dim=0)

                elif 'gallery_distractor' in dts.keys():
                    # the gallery set is already composed of the queries
                    lib.LOGGER.info(f"Getting embeddings for the queries of the {split} set")
                    features, labels, relevances = self.get_embeddings(net, dts["query"])
                    lib.LOGGER.info(f"Getting embeddings for the gallery & distractor of the {split} set")
                    ref_features, ref_labels, _ = self.get_embeddings(net, dts["gallery_distractor"])
                    embeddings_come_from_same_source = True

            else:
                raise ValueError(f"Unknown type for dataset: {type(dts)}")

            logs[split] = self.batch_metrics(
                features,
                labels,
                embeddings_come_from_same_source,
                relevances,
                ref_features,
                ref_labels,
            )

        if self.overall_accuracy:
            overall_logs = overall_accuracy_hook(logs)
            logs["test_overall"] = overall_logs

        return logs

    def __repr__(self):
        repr = (
            f"{self.__class__.__name__}(\n"
            f"    compute_for_hierarchy_levels={self.compute_for_hierarchy_levels},\n"
            f"    exclude={self.exclude},\n"
            f"    recall_rate={self.recall_rate},\n"
            f"    hard_ap_for_level={self.hard_ap_for_level},\n"
            f"    with_binary_asi={self.with_binary_asi},\n"
            f"    overall_accuracy={self.overall_accuracy},\n"
            f"    metric_batch_size={self.metric_batch_size},\n"
            f"    inference_batch_size={self.inference_batch_size},\n"
            f"    num_workers={self.num_workers},\n"
            f"    pin_memory={self.pin_memory},\n"
            f"    convert_to_cuda={self.convert_to_cuda},\n"
            f"    with_faiss={self.with_faiss},\n"
            ")"
        )
        return repr


@lib.get_set_random_state
def evaluate(
    net,
    dataset_dict,
    acc=None,
    epoch=None,
    **kwargs,
):
    if acc is None:
        acc = AccuracyCalculator(**kwargs)

    return acc.evaluate(
        net,
        dataset_dict,
        epoch=epoch,
    )
