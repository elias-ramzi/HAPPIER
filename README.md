# Hierarchical Average Precision Training for Pertinent Image Retrieval

This repo contains the official PyTorch implementation of the HAPPIER method as described in the ECCV 2022 paper: [Hierarchical Average Precision Training for Pertinent Image Retrieval](https://arxiv.org/abs/2207.04873).

### Suggested citation

Please consider citing our work:

```
@inproceedings{ramzi2022hierarchical,
  title={Hierarchical Average Precision Training for Pertinent Image Retrieval},
  author={Ramzi, Elias and Audebert, Nicolas and Thome, Nicolas and Rambour, Cl{\'e}ment and Bitot, Xavier},
  booktitle={European Conference on Computer Vision},
  pages={250--266},
  year={2022},
  organization={Springer}
}
```


![figure_methode](https://github.com/elias-ramzi/HAPPIER/blob/main/pictures/figure_method.png)

## Use HAPPIER

This will create a virtual environment and install the dependencies described in `requirements.txt`:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

WARNING: as of now this code does not work for newer version of `torch`. It only works with `torch==1.8.1`.


## Datasets

We use the following datasets for our paper:

- [iNaturalist-2018](https://github.com/visipedia/inat_comp/tree/master/2018#Data) with [splits](https://drive.google.com/file/d/1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98/view?usp=sharing)
- [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
- [DyML-datasets](https://onedrive.live.com/?authkey=%21AMLHa5h%2D56ZZL94&id=F4EF5F480284E1C2%21106&cid=F4EF5F480284E1C2)

Once extracted the code should work with the base structure of the datasets. You must precise the direction of the dataset to run an experiment:

```
dataset.data_dir=/Path/To/Your/Data/Stanford_Online_Products
```

For iNat you must put the split in the folder of the dataset: `Inaturalist/Inat_dataset_splits`.

You can also tweak the `lib/expand_path.py` function, as it is called for most path handling in the code.

### Add you dataset

When implementing your custom dataset it shoud herit from `BaseDataset`
```
from happier.datasets.base_dataset import BaseDataset


class CustomDataset(BaseDataset):
  HIERARCHY_LEVEL = L

  def __init__(data_dir, mode, transform, **kwargs):
    self.paths = ...
    self.labels = ...  # should a numpy array of ndim == 2

    super().__init__(**kwargs)  # this should be at the end.

```

Then add you `CustomDataset` to the `__init__.py` file of `datasets`.

```
from .custom_dataset import CustomDataset

__all__ = [
    'CustomDataset',
]
```

Finally you should create a config file `custom_dataset.yaml` in `happier/config/dataset`.

## Run the code

The code uses Hydra for the config. You can override arguments from command line or change a whole config. You can easily add other configs in happier/config.

Do not hesitate to create an issue if you have trouble understanding the configs, I will gladly answer you.

### iNaturalist

<details>
  <summary><b>iNat-base</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='0' python happier/run.py \
'experience.experiment_name=HAPPIER_iNat_base' \
'experience.log_dir=experiments/HAPPIER' \
experience.seed=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1] \
experience.warmup_step=5 \
optimizer=inat \
model=resnet_ln \
transform=inat \
dataset=inat_base \
loss=HAPPIER_inat
```

</details>

<details>
  <summary><b>iNat-full</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='0' python happier/run.py \
'experience.experiment_name=HAPPIER_iNat_full' \
'experience.log_dir=experiments/HAPPIER/' \
experience.seed=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1,2,3,4,5,6] \
experience.warmup_step=5 \
optimizer=inat \
model=resnet_ln \
transform=inat \
dataset=inat_full \
loss=HAPPIER_inat
```

</details>

### Stanford Online Products

<details>
  <summary><b>SOP</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='0' python happier/run.py \
'experience.experiment_name=HAPPIER_SOP' \
'experience.log_dir=experiments/HAPPIER' \
experience.seed=0 \
experience.max_iter=100 \
experience.warmup_step=5 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1] \
optimizer=sop \
model=resnet_ln \
transform=sop \
dataset=sop \
loss=HAPPIER_SOP
```

</details>


### Dynamic Metric Learning
<details>
  <summary><b>DyML-Vehicle</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='0' python happier/run.py \
'experience.experiment_name=HAPPIER_dyml_vehicle' \
'experience.log_dir=experiments/HAPPIER' \
experience.seed=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0] \
experience.accuracy_calculator.overall_accuracy=True \
experience.accuracy_calculator.exclude=[NDCG,H-AP] \
experience.accuracy_calculator.recall_rate=[10,20] \
experience.accuracy_calculator.with_binary_asi=True \
optimizer=dyml \
model=dyml_resnet34 \
transform=dyml \
dataset=dyml_vehicle \
loss=HAPPIER
```

</details>

<details>
  <summary><b>DyML-Animal</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='2' python happier/run.py \
'experience.experiment_name=HAPPIER_dyml_animal' \
'experience.log_dir=experiments/HAPPIER' \
experience.seed=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0] \
experience.accuracy_calculator.overall_accuracy=True \
experience.accuracy_calculator.exclude=[NDCG,H-AP] \
experience.accuracy_calculator.recall_rate=[10,20] \
experience.accuracy_calculator.with_binary_asi=True \
optimizer=dyml \
model=dyml_resnet34 \
transform=dyml \
dataset=dyml_animal \
loss=HAPPIER_5
```

</details>

<details>
  <summary><b>DyML-Product</b></summary><br/>

```
CUDA_VISIBLE_DEVICES='1' python happier/run.py \
'experience.experiment_name=HAPPIER_dyml_product' \
'experience.log_dir=experiments/HAPPIER' \
experience.seed=0 \
experience.max_iter=20 \
experience.warmup_step=5 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0,1,2] \
experience.accuracy_calculator.overall_accuracy=True \
experience.accuracy_calculator.exclude=[NDCG,H-AP] \
experience.accuracy_calculator.recall_rate=[10,20] \
experience.accuracy_calculator.with_binary_asi=True \
optimizer=dyml_product \
model=dyml_resnet34_product \
transform=dyml \
dataset=dyml_product \
loss=HAPPIER_product
```

</details>


## Resources

Links to repo with useful features used for this code:

- Hydra: https://github.com/facebookresearch/hydra
- ROADMAP: https://github.com/elias-ramzi/ROADMAP
- NSM: https://github.com/azgo14/classification_metric_learning
- PyTorch: https://github.com/pytorch/pytorch
- Pytorch Metric Learning (PML): https://github.com/KevinMusgrave/pytorch-metric-learning


## TODO LIST

- [x] Add instruction to reproduce all experiments
- [ ] Make H-AP easier to use outside this repository
- [ ] Clean H-AP loss code
- [ ] Create paper with code badge
