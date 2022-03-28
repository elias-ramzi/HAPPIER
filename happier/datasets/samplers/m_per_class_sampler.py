"""
adapted from :
https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/datasets.py
"""
import copy

import numpy as np
from pytorch_metric_learning import samplers

import happier.lib as lib


def flatten(list_):
    return [item for sublist in list_ for item in sublist]


class MPerClassSampler:
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class=4,
        hierarchy_level=0,
    ):
        assert samples_per_class > 1
        assert batch_size % samples_per_class == 0

        labels = dataset.labels[:, hierarchy_level]
        self.image_dict = {cl: [] for cl in set(labels)}
        for idx, cl in enumerate(labels):
            self.image_dict[cl].append(idx)

        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        self.reshuffle()

    def __iter__(self,):
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class}\n)"
        )

    def reshuffle(self):
        lib.LOGGER.info("Shuffling data")
        image_dict = copy.deepcopy(self.image_dict)
        for sub in image_dict:
            np.random.shuffle(image_dict[sub])

        classes = [*image_dict]
        np.random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (len(batch) < self.batch_size/self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:]

            if len(batch) == self.batch_size/self.samples_per_class:
                batch = flatten(batch)
                np.random.shuffle(batch)
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        np.random.shuffle(total_batches)
        self.batches = total_batches


class PMLMPerClassSampler(samplers.MPerClassSampler):
    """
    This sampler is almost the same as the one above (the sampling
    method is not exactly the same). But it allows to choose
    how many batches are sampled.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_iter,
        samples_per_class=4,
        hierarchy_level=0,
    ):
        super().__init__(
            dataset.labels[:, hierarchy_level],
            m=samples_per_class,
            batch_size=batch_size,
            length_before_new_iter=num_iter*batch_size,
        )

        self.samples_per_class = samples_per_class
        self.num_iter = num_iter
        self.reshuffle()

    def __iter__(self,):
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    num_iter={self.num_iter},\n"
            ")"
        )

    def reshuffle(self,):
        lib.LOGGER.info("Shuffling data")
        idxs = list(super().__iter__())
        n = len(idxs)
        batches = []
        for i in range(n // self.batch_size):
            batches.append(idxs[:self.batch_size])
            del idxs[:self.batch_size]

        np.random.shuffle(batches)
        self.batches = batches
