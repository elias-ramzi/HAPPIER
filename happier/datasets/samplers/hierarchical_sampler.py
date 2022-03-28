from pytorch_metric_learning.samplers import HierarchicalSampler as PMLHierarchicalSampler

import happier.lib as lib


class HierarchicalSampler(PMLHierarchicalSampler):

    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_tuple=4,
        super_classes_per_batch=2,
        inner_label=0,
        outer_label=1,
        restrict_number_of_batches=None,
    ):
        self.restrict_number_of_batches = restrict_number_of_batches

        super().__init__(
            dataset.labels,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            batches_per_super_tuple=batches_per_super_tuple,
            super_classes_per_batch=super_classes_per_batch,
            inner_label=inner_label,
            outer_label=outer_label,
        )

    def reshuffle(self,):
        lib.LOGGER.info("Shuffling data")
        super().reshuffle()
        if self.restrict_number_of_batches is not None:
            # batches are already shuffled
            self.batches = self.batches[:self.restrict_number_of_batches]

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_tuple={self.batches_per_super_tuple},\n"
            f"    super_classes_per_batch={self.super_classes_per_batch},\n"
            f"    restrict_number_of_batches={self.restrict_number_of_batches},\n"
            ")"
        )
