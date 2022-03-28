from os.path import join

import numpy as np
import pandas as pd

import happier.lib as lib
from happier.datasets.base_dataset import BaseDataset


class DyMLDataset(BaseDataset):
    HIERARCHY_LEVEL = 3

    def __init__(
        self,
        data_dir,
        mode,
        transform=None,
        use_custom_splits=False,
        seed=0,
        inner=1,
        outer=2,
        **kwargs
    ):
        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform
        self.use_custom_splits = use_custom_splits

        assert isinstance(self.use_custom_splits, bool)

        name = self.data_dir.split('/')[1:]
        if not name[-1]:
            name.remove("")
        name = name[-1].split('_')[-1]
        self.__class__.__name__ = f"DyML{name.title()}"

        if mode == 'train':
            if self.use_custom_splits:
                assert inner <= outer
                db = lib.load_json(join(self.data_dir, f"dyml_{name}_train_seed={seed}_inner={inner}_outer={outer}.json"))
                # sort with the number of the picture to ensure consistency
                self.paths = sorted(list(db.keys()), key=lambda pth: int(pth.split('/')[-1].split('.')[0]))
                labels = []
                for pth in self.paths:
                    labels.append(db[pth])
                labels = np.stack(labels, axis=0)
                self.paths = [lib.expand_path(pth) for pth in self.paths]

                self.CACHE_FILE = f"CUSTOM_seed={seed}_inner={inner}_outer={outer}_" + BaseDataset.CACHE_FILE

            else:
                table = pd.read_csv(join(self.data_dir, 'train', 'label.csv'))
                paths = table["fname"].tolist()
                self.paths = [join(self.data_dir, 'train', 'imgs', x) for x in paths]

                labels = table[[" fine_id0.jpg", " middle_id", " coarse_id"]].to_numpy()

            self.labels = lib.set_labels_to_range(labels)

        elif mode.startswith("test"):
            if self.use_custom_splits:
                assert inner <= outer
                db = lib.load_json(join(self.data_dir, f"dyml_{name}_test_seed={seed}_inner={inner}_outer={outer}.json"))
                # sort with the number of the picture to ensure consistency
                self.paths = sorted(list(db.keys()), key=lambda pth: int(pth.split('/')[-1].split('.')[0]))
                labels = []
                for pth in self.paths:
                    labels.append(db[pth])
                self.labels = np.stack(labels, axis=0)
                self.paths = [lib.expand_path(pth) for pth in self.paths]

                self.CACHE_FILE = f"CUSTOM_seed={seed}_inner={inner}_outer={outer}_" + BaseDataset.CACHE_FILE

            else:
                # mode is for example 'test_query_fine'
                _, type, granularity = mode.split("_")
                assert type in ['query', 'gallery']
                assert granularity in ['fine', 'middle', 'coarse']
                table = pd.read_csv(join(self.data_dir, f'bmk_{granularity}', f'{type}.csv'))
                paths = table["fname"].tolist()
                labels = table[" labels0.jpg"].to_numpy().reshape(-1, 1)

                self.paths = [join(self.data_dir, f"bmk_{granularity}", f"{type}", x) for x in paths]
                self.labels = labels

        else:
            raise ValueError(f"Unknown mode: {mode}")

        super().__init__(**kwargs)

    @property
    def my_sub_repr(self,):
        return f"    use_custom_splits={self.use_custom_splits},\n"


class DyMLProduct(BaseDataset):
    HIERARCHY_LEVEL = 3

    def __init__(
        self,
        data_dir,
        mode,
        transform=None,
        use_custom_splits=False,
        seed=0,
        inner=1,
        outer=2,
        **kwargs,
    ):

        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform
        self.use_custom_splits = use_custom_splits

        if self.use_custom_splits:
            lib.LOGGER.warning("You should use DyMLDataset for the product dataset and use_custom_splits=True")

        if mode == 'train':
            if use_custom_splits:
                db = lib.load_json(join(self.data_dir, f'dyml_product_train_seed={seed}_inner={inner}_outer={outer}.json'))
                # sort with the number of the picture to ensure consistency
                self.paths = sorted(list(db.keys()), key=lambda pth: int(pth.split('/')[-1].split('.')[0]))
                labels = []
                for pth in self.paths:
                    labels.append(db[pth])
                labels = np.stack(labels, axis=0)
                self.paths = [lib.expand_path(pth) for pth in self.paths]

                self.CACHE_FILE = f"CUSTOM_seed={seed}_inner={inner}_outer={outer}_" + BaseDataset.CACHE_FILE

            else:
                table = pd.read_csv(join(self.data_dir, 'train', 'label.csv'))
                paths = table["fname"].tolist()
                self.paths = [join(self.data_dir, 'train', 'imgs', x) for x in paths]

                labels = table[[" fine_id0.jpg", " middle_id", " coarse_id"]].to_numpy()

            self.labels = lib.set_labels_to_range(labels)

        elif mode == 'test':
            if use_custom_splits:
                # sort with the number of the picture to ensure consistency
                db = lib.load_json(join(self.data_dir, f'dyml_product_test_seed={seed}_inner={inner}_outer={outer}.json'))
                self.paths = sorted(list(db.keys()), key=lambda pth: int(pth.split('/')[-1].split('.')[0]))
                labels = []
                for pth in self.paths:
                    labels.append(db[pth])
                self.labels = np.stack(labels, axis=0)
                self.paths = [lib.expand_path(pth) for pth in self.paths]

                self.CACHE_FILE = f"CUSTOM_seed={seed}_inner={inner}_outer={outer}_" + BaseDataset.CACHE_FILE

            else:
                table = pd.read_csv(join(self.data_dir, "mini-bmk_all_in_one", 'label.csv'))
                paths = table["fname"].tolist()
                self.paths = [join(self.data_dir, "mini-bmk_all_in_one", "imgs", x) for x in paths]

                self.labels = table[[" fine_id0.jpg", " middle_id", " coarse_id"]].to_numpy()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        super().__init__(**kwargs)

    @property
    def my_sub_repr(self,):
        return f"    use_custom_splits={self.use_custom_splits},\n"
