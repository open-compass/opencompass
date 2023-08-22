# OTTER: Multi-modal In-context Instruction Tuning.

### Prepare the environment

```sh
cd opencompass/multimodal/models/otter
git clone https://github.com/Luodian/Otter.git
```

Then create a new conda environment and prepare the environement according to this [doc](https://github.com/Luodian/Otter)

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