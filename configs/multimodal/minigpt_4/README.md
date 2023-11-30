# MiniGPT-4

### Prepare the environment

```sh
cd opencompass/multimodal/models/minigpt_4
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
```

Then prepare the environment according to this [doc](https://github.com/Vision-CAIR/MiniGPT-4)

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
