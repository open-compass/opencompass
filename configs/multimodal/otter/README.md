# OTTER: Multi-modal In-context Instruction Tuning.

### Prepare the environment

```sh
pip install otter_ai
```

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