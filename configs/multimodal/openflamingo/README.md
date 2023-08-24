# OpenFlamingo

### Prepare the environment

Install [MMPretrain](https://github.com/open-mmlab/mmpretrain) according to this [doc](https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation)

### Start evaluation

#### Slurm

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval --slurm -p $PARTITION
```

#### PyTorch

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval
```