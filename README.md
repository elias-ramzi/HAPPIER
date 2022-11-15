# Hierarchical Average Precision Training for Pertinent Image Retrieval

This repo contains the official PyTorch implementation of the HAPPIER method as described in the ECCV 2022 paper: [Hierarchical Average Precision Training for Pertinent Image Retrieval](https://arxiv.org/abs/2207.04873).

### Suggested citation

Please consider citing our work:

```
@inproceedings{ramzi2022hierarchical,
  title={Hierarchical Average Precision Training for Pertinent Image Retrieval},
  author={Ramzi, Elias and Audebert, Nicolas and Thome, Nicolas and Rambour, Cl{\'e}ment and Bitot, Xavier},
  booktitle={Proceedings of the IEEE/CVF European Conference on Computer Vision},
  year={2022}
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

- [DyML-datasets](https://onedrive.live.com/?authkey=%21AMLHa5h%2D56ZZL94&id=F4EF5F480284E1C2%21106&cid=F4EF5F480284E1C2)
- [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
- [INaturalist-2018](https://github.com/visipedia/inat_comp/tree/master/2018#Data) with [splits](https://drive.google.com/file/d/1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98/view?usp=sharing)

Once extracted the code should work with the base structure of the datasets. You must precise the direction of the dataset to run an experiment:

```
dataset.data_dir=/Path/To/Your/Data/Stanford_Online_Products
```

For iNat you must put the split in the folder of the dataset: `Inaturalist/Inat_dataset_splits`.

You can also tweak the lib/expand_path.py function, as it is called for most path handling in the code.


## Run the code

The code uses Hydra for the config. You can override arguments from command line or change a whole config. You can easily add other configs in happier/config.

Do not hesitate to create an issue if you have trouble understanding the configs. I will gladly answer you.

Instructions coming soon!

### DyML-Vehicle
CUDA_VISIBLE_DEVICES='0' python happier/run.py \
'experience.experiment_name=HAPPIER_dyml_vehicle' \
'experience.log_dir=experiments/HAPPIER' \
experience.seed=0 \
experience.accuracy_calculator.compute_for_hierarchy_levels=[0] \
experience.accuracy_calculator.overall_accuracy=True \
experience.accuracy_calculator.exclude=[NDCG,H-AP] \
experience.accuracy_calculator.recall_rate=[10,20] \
experience.accuracy_calculator.with_binary_asi=True \
optimizer=dyml_animal \
model=dyml_resnet34 \
transform=inat \
dataset=dyml_vehicle \
loss=HAPPIER


## Resources

Links to repo with useful features used for this code:

- Hydra: https://github.com/facebookresearch/hydra
- ROADMAP: https://github.com/elias-ramzi/ROADMAP
- NSM: https://github.com/azgo14/classification_metric_learning
- PyTorch: https://github.com/pytorch/pytorch
- Pytorch Metric Learning (PML): https://github.com/KevinMusgrave/pytorch-metric-learning


## TODO LIST

- [ ] Add instruction to reproduce all experiments
- [ ] Make H-AP easier to use outside this repository
- [ ] Clean H-AP loss code
