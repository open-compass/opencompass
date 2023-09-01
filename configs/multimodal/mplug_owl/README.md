# MplugOwl

### Prepare the environment

```sh
cd opencompass/multimodal/models/mplug_owl
git clone https://github.com/X-PLUG/mPLUG-Owl.git
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